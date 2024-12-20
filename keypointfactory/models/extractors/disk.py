from pathlib import Path

import torch
from omegaconf import OmegaConf

from ...geometry.depth import simple_project, unproject
from ...geometry.epipolar import (
    T_to_F,
    asymm_epipolar_distance_all,
    relative_pose_error,
)
from ...geometry.homography import homography_corner_error, warp_points_torch
from ...robust_estimators import load_estimator
from ...settings import DATA_PATH, TRAINING_PATH
from ..base_model import BaseModel
from ..utils.misc import distance_matrix, pad_and_stack, select_on_last, tile
from ..utils.unet import Unet


def point_distribution(logits):
    proposal_dist = torch.distributions.Categorical(logits=logits)
    proposals = proposal_dist.sample()
    proposal_logp = proposal_dist.log_prob(proposals)

    accept_logits = select_on_last(logits, proposals).squeeze(-1)

    accept_dist = torch.distributions.Bernoulli(logits=accept_logits)
    accept_samples = accept_dist.sample()
    accept_logp = accept_dist.log_prob(accept_samples)
    accept_mask = accept_samples == 1

    logp = proposal_logp + accept_logp

    return proposals, accept_mask, logp


def reproject_homography(kpts, H, h, w, inverse):
    kptsw = warp_points_torch(kpts, H, inverse)

    valid = (
        (0 <= kptsw[:, :, 0])
        & (kptsw[:, :, 0] < w)
        & (0 <= kptsw[:, :, 1])
        & (kptsw[:, :, 1] < h)
    )

    nkpts = torch.full(
        kpts.shape,
        fill_value=float("NaN"),
        device=kpts.device,
        dtype=kpts.dtype,
    )
    nkpts[valid] = kptsw[valid]

    return nkpts


class ConsistentMatchDistribution:
    def __init__(
        self,
        pred,
        inverse_T,
    ):
        self.inverse_T = inverse_T
        descriptors0 = pred["descriptors0"]
        descriptors1 = pred["descriptors1"]

        distances = distance_matrix(
            descriptors0,
            descriptors1,
        ).to(inverse_T.device)
        if distances.dim() == 2:
            distances = distances.unsqueeze(0)

        affinity = -inverse_T * distances
        self._cat_I = torch.distributions.Categorical(logits=affinity)
        self._cat_T = torch.distributions.Categorical(logits=affinity.transpose(1, 2))

        self._dense_logp = None
        self._dense_p = None

    def dense_p(self):
        if self._dense_p is None:
            self._dense_p = self._cat_I.probs * self._cat_T.probs.transpose(1, 2)

        return self._dense_p

    def dense_logp(self):
        if self._dense_logp is None:
            self._dense_logp = self._cat_I.logits + self._cat_T.logits.transpose(1, 2)

        return self._dense_logp

    def _select_cycle_consistent(self, left, right):
        indexes = torch.arange(left.shape[0], device=left.device)
        cycle_consistent = right[left] == indexes

        paired_left = left[cycle_consistent]

        return torch.stack(
            [
                right[paired_left],
                paired_left,
            ],
            dim=0,
        )

    def sample(self):
        samples_I = self._cat_I.sample()
        samples_T = self._cat_T.sample()

        return self._select_cycle_consistent(samples_I, samples_T)

    def mle(self):
        maxes_I = self._cat_I.logits.argmax(dim=1)
        maxes_T = self._cat_T.logits.argmax(dim=1)

        return self._select_cycle_consistent(maxes_I, maxes_T)


class ConsistentMatcher(torch.nn.Module):
    def __init__(self, inverse_T=1.0):
        super(ConsistentMatcher, self).__init__()
        self.inverse_T = torch.nn.Parameter(
            torch.tensor(inverse_T, dtype=torch.float32)
        )

    def extra_repr(self):
        return f"inverse_T={self.inverse_T.item()}"

    def match_pair(self, pred):
        return ConsistentMatchDistribution(pred, self.inverse_T)


class CycleMatcher:
    def match_by_homography(self, data, pred):
        kpts0 = pred["keypoints0"]
        kpts1 = pred["keypoints1"]

        H_0to1 = data["H_0to1"]

        kpts0_r = reproject_homography(
            kpts0, H_0to1, *data["view1"]["image"].shape[2:], False
        )
        kpts1_r = reproject_homography(
            kpts1, H_0to1, *data["view0"]["image"].shape[2:], True
        )

        diff0 = kpts1_r[:, None, :, :] - kpts0[:, :, None, :]
        diff1 = kpts0_r[:, :, None, :] - kpts1[:, None, :, :]

        dist0 = torch.norm(diff0, dim=-1)
        dist1 = torch.norm(diff1, dim=-1)
        dist = torch.max(dist0, dist1)
        mask_visible = ~torch.isnan(dist)
        inf = dist.new_tensor(float("inf"))
        dist = torch.where(mask_visible, dist, inf)

        min0 = dist.min(-1).indices
        min1 = dist.min(-2).indices
        ismin0 = torch.zeros(dist.shape, dtype=torch.bool, device=dist.device)
        ismin1 = ismin0.clone()

        ismin0.scatter_(-1, min0.unsqueeze(-1), value=1)
        ismin1.scatter_(-2, min1.unsqueeze(-2), value=1)
        positive = ismin0 & ismin1
        validm0 = positive.any(-1)
        validm1 = positive.any(-2)

        unmatched0 = min0.new_tensor(-1)
        m0 = torch.where(validm0, min0, unmatched0)
        m1 = torch.where(validm1, min1, unmatched0)

        matches = []
        for b in range(kpts0.shape[0]):
            cyclical0 = (
                m1[b][m0[b][validm0[b]]]
                == torch.arange(m0[b].shape[0], device=m0.device)[validm0[b]]
            )

            matches.append(
                torch.stack(
                    [
                        torch.arange(m0.shape[1], device=m0.device)[validm0[b]][
                            cyclical0
                        ],
                        m0[b][validm0[b]][cyclical0],
                    ]
                )
            )
        matches = pad_and_stack(matches, None, -1, mode="constant", constant=-1)
        return matches.transpose(1, 2)

    def match_by_depth(self, data, pred):
        kpts0 = pred["keypoints0"]  # [B, N, 2]
        kpts1 = pred["keypoints1"]  # [B, M, 2]

        depth0 = data["view0"]["depth"]
        depth1 = data["view1"]["depth"]
        cam0 = data["view0"]["camera"]
        cam1 = data["view1"]["camera"]
        T0 = data["view0"]["T_w2cam"]
        T1 = data["view1"]["T_w2cam"]

        kpts0_r = simple_project(unproject(kpts0, depth0, cam0, T0), cam1, T1)
        kpts1_r = simple_project(unproject(kpts1, depth1, cam1, T1), cam0, T0)

        diff0 = kpts1_r[:, None, :, :] - kpts0[:, :, None, :]
        diff1 = kpts0_r[:, :, None, :] - kpts1[:, None, :, :]

        dist0 = torch.norm(diff0, dim=-1)
        dist1 = torch.norm(diff1, dim=-1)
        dist = torch.max(dist0, dist1)
        mask_visible = ~torch.isnan(dist)
        inf = dist.new_tensor(float("inf"))
        dist = torch.where(mask_visible, dist, inf)

        min0 = dist.min(-1).indices
        min1 = dist.min(-2).indices
        ismin0 = torch.zeros(dist.shape, dtype=torch.bool, device=dist.device)
        ismin1 = ismin0.clone()

        ismin0.scatter_(-1, min0.unsqueeze(-1), value=1)
        ismin1.scatter_(-2, min1.unsqueeze(-2), value=1)
        positive = ismin0 & ismin1
        validm0 = positive.any(-1)
        validm1 = positive.any(-2)

        unmatched0 = min0.new_tensor(-1)
        m0 = torch.where(validm0, min0, unmatched0)
        m1 = torch.where(validm1, min1, unmatched0)

        matches = []
        for b in range(kpts0.shape[0]):
            cyclical0 = (
                m1[b][m0[b][validm0[b]]]
                == torch.arange(m0[b].shape[0], device=m0.device)[validm0[b]]
            )

            matches.append(
                torch.stack(
                    [
                        torch.arange(m0.shape[1], device=m0.device)[validm0[b]][
                            cyclical0
                        ],
                        m0[b][validm0[b]][cyclical0],
                    ]
                )
            )
        matches = pad_and_stack(matches, None, -1, mode="constant", constant=-1)
        return matches.transpose(1, 2)

    def match_by_descriptors(self, pred):
        descriptors0 = pred["descriptors0"]
        descriptors1 = pred["descriptors1"]

        distances = distance_matrix(
            descriptors0,
            descriptors1,
        ).to(descriptors0.device)
        if distances.dim() == 2:
            distances = distances.unsqueeze(0)

        n_amin = torch.argmin(distances, dim=-1)
        m_amin = torch.argmin(distances, dim=-2)

        nnnn = m_amin.gather(1, n_amin)

        n_ix = torch.arange(distances.shape[-2], device=distances.device)
        matches = []
        for i in range(nnnn.shape[0]):
            mask = nnnn[i] == n_ix
            matches.append(
                torch.stack(
                    [torch.nonzero(mask, as_tuple=False)[:, 0], n_amin[i][mask]], dim=0
                )
            )
        matches = pad_and_stack(matches, None, -1, mode="constant", constant=-1)
        return matches.transpose(1, 2)


def classify_by_homography(data, pred, threshold=2.0):
    kpts0 = pred["keypoints0"]
    kpts1 = pred["keypoints1"]

    H_0to1 = data["H_0to1"]

    kpts0_r = reproject_homography(
        kpts0, H_0to1, *data["view1"]["image"].shape[2:], False
    )
    kpts1_r = reproject_homography(
        kpts1, H_0to1, *data["view0"]["image"].shape[2:], True
    )

    diff0 = kpts1_r[:, None, :, :] - kpts0[:, :, None, :]
    diff1 = kpts0_r[:, :, None, :] - kpts1[:, None, :, :]

    close0 = torch.norm(diff0, p=2, dim=-1) < threshold
    close1 = torch.norm(diff1, p=2, dim=-1) < threshold

    return close0 & close1


def classify_by_epipolar(data, pred, threshold=2.0):
    F_0_1 = T_to_F(data["view0"]["camera"], data["view1"]["camera"], data["T_0to1"])
    F_1_0 = T_to_F(data["view1"]["camera"], data["view0"]["camera"], data["T_1to0"])

    kpts0 = pred["keypoints0"]
    kpts1 = pred["keypoints1"]

    epi_0_1 = asymm_epipolar_distance_all(kpts0, kpts1, F_0_1).abs()
    epi_1_0 = asymm_epipolar_distance_all(kpts1, kpts0, F_1_0).abs()

    if epi_0_1.dim() == 2:
        epi_0_1 = epi_0_1.unsqueeze(0)
    if epi_1_0.dim() == 2:
        epi_1_0 = epi_1_0.unsqueeze(0)

    return (epi_0_1 < threshold) & (epi_1_0 < threshold).transpose(1, 2)


def classify_by_depth(data, pred, threshold=2.0):
    kpts0 = pred["keypoints0"]  # [B, N, 2]
    kpts1 = pred["keypoints1"]  # [B, M, 2]

    depth0 = data["view0"]["depth"]
    depth1 = data["view1"]["depth"]
    cam0 = data["view0"]["camera"]
    cam1 = data["view1"]["camera"]
    T0 = data["view0"]["T_w2cam"]
    T1 = data["view1"]["T_w2cam"]

    kpts0_r = simple_project(unproject(kpts0, depth0, cam0, T0), cam1, T1)
    kpts1_r = simple_project(unproject(kpts1, depth1, cam1, T1), cam0, T0)

    diff0 = kpts1_r[:, None, :, :] - kpts0[:, :, None, :]
    diff1 = kpts0_r[:, :, None, :] - kpts1[:, None, :, :]

    close0 = torch.norm(diff0, p=2, dim=-1) < threshold
    close1 = torch.norm(diff1, p=2, dim=-1) < threshold

    return close0 & close1


def epipolar_reward(data, pred, threshold=2.0, lm_tp=1.0, lm_fp=-0.25):
    good = classify_by_epipolar(data, pred, threshold)
    return lm_tp * good + lm_fp * (~good)


def depth_reward(data, pred, threshold=2.0, lm_tp=1.0, lm_fp=-0.25):
    epi_bad = ~classify_by_epipolar(data, pred, threshold)
    good_pairs = classify_by_depth(data, pred, threshold)

    return lm_tp * good_pairs + lm_fp * epi_bad


def homography_reward(data, pred, threshold=2.0, lm_tp=1.0, lm_fp=-0.25):
    good_pairs = classify_by_homography(data, pred, threshold)

    return lm_tp * good_pairs + lm_fp * ~good_pairs


class DISK(BaseModel):
    default_conf = {
        "window_size": 8,
        "nms_radius": 2,  # matches with disk nms radius of 5
        "max_num_keypoints": None,
        "force_num_keypoints": False,
        "pad_if_not_divisible": True,
        "detection_threshold": 0.005,
        "desc_dim": 128,
        "weights": None,
        "reward": "depth",
        "arch": {
            "kernel_size": 5,
            "down": [16, 32, 64, 64, 64],
            "up": [64, 64, 64],
            "gate": "PReLU",
            "norm": "InstanceNorm2d",
            "upsample": "TrivialUpsample",
            "downsample": "TrivialDownsample",
            "down_block": "ThinDownBlock",  # second option is DownBlock
            "up_block": "ThinUpBlock",  # second option is UpBlock
            # "dropout": False, not used yet
            "bias": True,
            "padding": True,
            "train_invT": False,
        },
        "loss": {
            "reward_threshold": 1.5,
            "lambda_tp": 1,
            "lambda_fp": -0.25,
            "lambda_kp": -0.001,
        },
        "estimator": {"name": "degensac", "ransac_th": 1.0},
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        self.set_initialized()

        self.conf = OmegaConf.merge(
            self.conf,
            OmegaConf.create(
                {
                    "arch": {
                        "down": OmegaConf.to_container(self.conf.arch.down),
                        "up": OmegaConf.to_container(self.conf.arch.up)
                        + [self.conf.desc_dim + 1],
                    }
                }
            ),
        )
        self.unet = Unet(
            in_features=3,
            conf=self.conf,
        )

        self.train_matcher = ConsistentMatcher(inverse_T=15.0)
        self.train_matcher.requires_grad_(self.conf.arch.train_invT)

        state_dict = None
        if conf.weights:
            if Path(conf.weights).exists():
                print(f"Loading weights from {conf.weights}")
                state_dict = torch.load(conf.weights, map_location="cpu")
            elif (Path(DATA_PATH) / conf.weights).exists():
                print(f"Loading weights from {Path(DATA_PATH) / conf.weights}")
                state_dict = torch.load(
                    Path(DATA_PATH) / conf.weights, map_location="cpu"
                )
            elif (Path(TRAINING_PATH) / conf.weights).exists():
                print(f"Loading weights from {Path(TRAINING_PATH) / conf.weights}")
                state_dict = torch.load(
                    Path(TRAINING_PATH) / conf.weights, map_location="cpu"
                )
            else:
                raise RuntimeError(f"Could not find weights at {conf.weights}")

        if state_dict:
            if "disk" in state_dict or "extractor" in state_dict:
                print("Detected original disk repo weights...")
                model_sd = {}
                key = "disk" if "disk" in state_dict else "extractor"
                for k, v in state_dict[key].items():
                    parts = k.split(".")
                    parts[3] = "sequence"
                    parts[4] = (
                        "0"
                        if (parts[1] == "path_down" and parts[2]) == "0"
                        else ("1" if parts[4] == "1" else "2")
                    )
                    model_sd[".".join(parts)] = v

                missing_keys, unexpected_keys = self.load_state_dict(
                    model_sd, strict=False
                )
            else:
                model_sd = {}
                print("Keypointfactory repo weights...")
                for k, v in state_dict["model"].items():
                    model_sd[k.replace("extractor.", "")] = v
                missing_keys, unexpected_keys = self.load_state_dict(
                    model_sd, strict=False
                )
            print(missing_keys)
            print(unexpected_keys)

        self.val_matcher = CycleMatcher()
        # Turns training off for matcher
        self.lm_tp = self.conf.loss.lambda_tp
        self.lm_fp = self.conf.loss.lambda_fp
        self.lm_kp = self.conf.loss.lambda_kp

    def _sample(self, heatmaps):
        v = self.conf.window_size
        device = heatmaps.device
        b, _, h, w = heatmaps.shape

        assert h % v == 0
        assert w % v == 0

        tiled = tile(heatmaps, self.conf.window_size).squeeze(1)

        proposals, accept_mask, logp = point_distribution(tiled)

        cgrid = torch.stack(
            torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
            )[::-1],
            dim=0,
        ).unsqueeze(0)
        cgrid_tiled = tile(cgrid, self.conf.window_size)

        xys = select_on_last(
            cgrid_tiled.repeat(b, 1, 1, 1, 1), proposals.unsqueeze(1).repeat(1, 2, 1, 1)
        ).permute(0, 2, 3, 1)

        keypoints = []
        scores = []

        for i in range(b):
            keypoints.append(xys[i][accept_mask[i]])
            scores.append(logp[i][accept_mask[i]])

        return keypoints, scores

    def _nms(self, heatmaps):
        heatmaps = heatmaps.squeeze(1)

        ixs = torch.nn.functional.max_pool2d(
            heatmaps,
            kernel_size=2 * self.conf.nms_radius + 1,
            stride=1,
            padding=self.conf.nms_radius,
            return_indices=True,
        )[1]

        h, w = heatmaps.shape[1:]
        coords = torch.arange(h * w, device=heatmaps.device).reshape(1, h, w)
        nms = ixs == coords

        if self.conf.detection_threshold is not None:
            nms = nms & (heatmaps > self.conf.detection_threshold)

        keypoints = []
        scores = []

        for i in range(heatmaps.shape[0]):
            keypoints.append(torch.flip(nms[i].nonzero(as_tuple=False), (1,)))
            scores.append(heatmaps[i][nms[i]])

        return keypoints, scores

    def get_heatmap_and_descriptors(self, data):
        images = data["image"]
        if self.conf.pad_if_not_divisible:
            h, w = images.shape[2:]
            pd_h = 16 - h % 16 if h % 16 > 0 else 0
            pd_w = 16 - w % 16 if w % 16 > 0 else 0
            images = torch.nn.functional.pad(images, (0, pd_w, 0, pd_h), value=0.0)

        output = self.unet(images)
        descs = output[:, : self.conf.desc_dim]
        heatmaps = output[:, self.conf.desc_dim :]

        return heatmaps, descs

    def _forward(self, data):
        images = data["image"]
        if self.conf.pad_if_not_divisible:
            h, w = images.shape[2:]
            pd_h = 16 - h % 16 if h % 16 > 0 else 0
            pd_w = 16 - w % 16 if w % 16 > 0 else 0
            images = torch.nn.functional.pad(images, (0, pd_w, 0, pd_h), value=0.0)

        output = self.unet(images)
        descs = output[:, : self.conf.desc_dim]
        heatmaps = output[:, self.conf.desc_dim :]

        if self.training:
            # Use sampling during training
            points, logps = self._sample(heatmaps)
        else:
            # Use NMS during evaluation
            points, logps = self._nms(heatmaps)

        keypoints = []
        scores = []
        descriptors = []

        for i, (point, logp) in enumerate(zip(points, logps)):
            if self.conf.max_num_keypoints is not None:
                n = min(self.conf.max_num_keypoints + 1, logp.numel())
                minus_threshold, _ = torch.kthvalue(-logp, n)
                mask = logp > -minus_threshold

                point = point[mask]
                logp = logp[mask]

            x, y = point.T
            desc = descs[i][:, y, x].T
            desc = torch.nn.functional.normalize(desc, dim=-1)

            keypoints.append(point)
            scores.append(logp)
            descriptors.append(desc)

        if self.conf.force_num_keypoints:
            # pad to target_length
            target_length = self.conf.max_num_keypoints
            keypoints = pad_and_stack(
                keypoints,
                target_length,
                -2,
                mode="zeros",
            )
            scores = pad_and_stack(scores, target_length, -1, mode="zeros")
            descriptors = pad_and_stack(descriptors, target_length, -2, mode="zeros")
        else:
            keypoints = torch.stack(keypoints, 0)
            scores = torch.stack(scores, 0)
            descriptors = torch.stack(descriptors, 0)

        return {
            "keypoints": keypoints.float(),
            "heatmap": heatmaps,
            "keypoint_scores": scores.float(),
            "descriptors": descriptors.float(),
        }

    def _pre_loss_callback(self, seed, epoch):
        if epoch == 0:
            ramp = 0
        elif epoch == 1:
            ramp = 0.1
        else:
            ramp = min(1.0, 0.1 + 0.2 * epoch)

        self.lm_fp = self.conf.loss.lambda_fp * ramp
        self.lm_kp = self.conf.loss.lambda_kp * ramp

        updated_inverse_T = torch.tensor(15.0 + 35.0 * min(1.0, 0.05 * epoch))
        # updated_inverse_T = torch.tensor(25.0 + epoch)

        self.train_matcher.inverse_T.copy_(updated_inverse_T)

    def loss(self, pred, data):
        if self.conf.reward == "depth":
            elementwise_reward = depth_reward(
                data,
                pred,
                threshold=self.conf.loss.reward_threshold,
                lm_tp=self.lm_tp,
                lm_fp=self.lm_fp,
            )
        elif self.conf.reward == "epipolar":
            elementwise_reward = epipolar_reward(
                data,
                pred,
                threshold=self.conf.loss.reward_threshold,
                lm_tp=self.lm_tp,
                lm_fp=self.lm_fp,
            )
        elif self.conf.reward == "homography":
            elementwise_reward = homography_reward(
                data,
                pred,
                threshold=self.conf.loss.reward_threshold,
                lm_tp=self.lm_tp,
                lm_fp=self.lm_fp,
            )
        else:
            raise ValueError(f"Unknown reward type {self.conf.reward}")

        match_dist = self.train_matcher.match_pair(pred)

        with torch.no_grad():
            sample_p = match_dist.dense_p()

        sample_logp = match_dist.dense_logp()

        logp0 = pred["keypoint_scores0"]
        logp1 = pred["keypoint_scores1"]

        kpts_logp = logp0[:, :, None] + logp1[:, None, :]

        sample_lp_flat = logp0.sum(dim=1) + logp1.sum(dim=1)

        sample_plogp = sample_p * (sample_logp + kpts_logp)

        reinforce = (elementwise_reward * sample_plogp).sum(dim=(-1, -2))

        kp_penalty = self.lm_kp * sample_lp_flat

        loss = -reinforce - kp_penalty

        losses = {
            "total": loss,
            "reinforce": reinforce,
            "kp_penalty": kp_penalty,
            "sample_lp": sample_lp_flat,
            "lm_kp": torch.tensor([self.lm_kp] * loss.shape[0], dtype=torch.float64),
            "lm_tp": torch.tensor([self.lm_tp] * loss.shape[0], dtype=torch.float64),
            "lm_fp": torch.tensor([self.lm_fp] * loss.shape[0], dtype=torch.float64),
            "n_kpts": torch.tensor(
                [logp0.shape[1]] * loss.shape[0],
                device=logp0.device,
                dtype=torch.float64,
            ),
        }
        del (
            logp0,
            logp1,
            sample_p,
            sample_logp,
            kpts_logp,
            sample_lp_flat,
            sample_plogp,
            reinforce,
            kp_penalty,
            loss,
        )

        if not self.training:
            if pred["keypoints0"].shape[-2] == 0 or pred["keypoints1"].shape[-2] == 0:
                zero = torch.zeros(
                    pred["keypoints0"].shape[0], device=pred["keypoints0"].device
                )
                metrics = {
                    "n_kpts": zero,
                    "n_pairs": zero,
                    "n_good": zero,
                    "n_bad": zero,
                    "prec": zero,
                    "reward": zero,
                }
            else:
                n_kpts = torch.tensor(
                    pred["keypoints0"].shape[-2] + pred["keypoints1"].shape[-2],
                    device=pred["keypoints0"].device,
                ).repeat(pred["keypoints0"].shape[0])

                def get_match_metrics(
                    kpts0, kpts1, M_gt, matches, image_size=None, estimate="relpose"
                ):
                    results = {}

                    valid_indices0 = (matches[:, :, 0] >= 0) & (
                        matches[:, :, 0] < kpts0.shape[1]
                    )
                    valid_indices1 = (matches[:, :, 1] >= 0) & (
                        matches[:, :, 1] < kpts1.shape[1]
                    )
                    kpts0 = kpts0.gather(
                        1,
                        matches[:, :, 0]
                        .unsqueeze(-1)
                        .masked_fill(~valid_indices0.unsqueeze(2), 0)
                        .repeat(1, 1, 2),
                    )
                    kpts1 = kpts1.gather(
                        1,
                        matches[:, :, 1]
                        .unsqueeze(-1)
                        .masked_fill(~valid_indices1.unsqueeze(2), 0)
                        .repeat(1, 1, 2),
                    )

                    results["n_pairs"] = valid_indices0.count_nonzero(dim=1)

                    if estimate == "relpose":
                        good = classify_by_epipolar(
                            data, {"keypoints0": kpts0, "keypoints1": kpts1}
                        )
                    elif estimate == "homography":
                        good = classify_by_homography(
                            data, {"keypoints0": kpts0, "keypoints1": kpts1}
                        )
                    good = good.diagonal(dim1=-2, dim2=-1)
                    bad = ~good

                    results["n_good"] = good.to(torch.int64).sum(dim=1)
                    results["n_bad"] = bad.to(torch.int64).sum(dim=1)
                    results["prec"] = results["n_good"] / (results["n_pairs"] + 1)

                    results["reward"] = (
                        self.lm_tp * results["n_good"]
                        + self.lm_fp * results["n_bad"]
                        + self.lm_kp * n_kpts
                    )

                    results["ransac_inl"] = torch.tensor([], device=kpts0.device)
                    results["ransac_inl%"] = torch.tensor([], device=kpts0.device)
                    for b in range(kpts0.shape[0]):
                        if estimate == "relpose":
                            results["rel_pose_error"] = torch.tensor(
                                [], device=kpts0.device
                            )
                            estimator = load_estimator(
                                "relative_pose", self.conf.estimator["name"]
                            )(self.conf.estimator)
                            data_ = {
                                "m_kpts0": kpts0[b].unsqueeze(0),
                                "m_kpts1": kpts1[b].unsqueeze(0),
                                "camera0": data["view0"]["camera"][b],
                                "camera1": data["view1"]["camera"][b],
                            }
                            est = estimator(data_)
                        elif estimate == "homography":
                            results["H_error"] = torch.tensor([], device=kpts0.device)
                            estimator = load_estimator(
                                "homography", self.conf.estimator["name"]
                            )(self.conf.estimator)
                            data_ = {
                                "m_kpts0": kpts0[b].unsqueeze(0),
                                "m_kpts1": kpts1[b].unsqueeze(0),
                            }
                            est = estimator(data_)

                        if not est["success"]:
                            if estimate == "relpose":
                                results["rel_pose_error"] = torch.cat(
                                    [
                                        results["rel_pose_error"],
                                        torch.tensor(
                                            [float("inf")], device=kpts0.device
                                        ),
                                    ]
                                )
                            elif estimate == "homography":
                                results["H_error"] = torch.cat(
                                    [
                                        results["H_error"],
                                        torch.tensor(
                                            [float("inf")], device=kpts0.device
                                        ),
                                    ]
                                )
                            results["ransac_inl"] = torch.cat(
                                [
                                    results["ransac_inl"],
                                    torch.tensor([0], device=kpts0.device),
                                ]
                            )
                            results["ransac_inl%"] = torch.cat(
                                [
                                    results["ransac_inl%"],
                                    torch.tensor([0.0], device=kpts0.device),
                                ]
                            )
                        else:
                            M = est["M_0to1"]
                            inl = est["inliers"]
                            if estimate == "relpose":
                                t_error, r_error = relative_pose_error(
                                    M_gt[b], M.R, M.t
                                )

                                results["rel_pose_error"] = torch.cat(
                                    [
                                        results["rel_pose_error"],
                                        max(r_error, t_error).unsqueeze(0),
                                    ]
                                )
                            elif estimate == "homography":
                                results["H_error"] = torch.cat(
                                    [
                                        results["H_error"],
                                        homography_corner_error(M, M_gt, image_size[b]),
                                    ]
                                )

                            results["ransac_inl"] = torch.cat(
                                [results["ransac_inl"], torch.sum(inl).unsqueeze(0)]
                            )
                            results["ransac_inl%"] = torch.cat(
                                [results["ransac_inl%"], torch.mean(inl).unsqueeze(0)]
                            )

                    return results

                metrics = {
                    "n_kpts": n_kpts,
                }

                if "depth" in data["view0"]:
                    desc_matches = self.val_matcher.match_by_descriptors(pred)
                    depth_matches = self.val_matcher.match_by_depth(data, pred)
                    kpts0 = pred["keypoints0"]
                    kpts1 = pred["keypoints1"]

                    desc_metrics = get_match_metrics(
                        kpts0,
                        kpts1,
                        data["T_0to1"],
                        desc_matches,
                        estimate="relpose",
                    )
                    depth_metrics = get_match_metrics(
                        kpts0,
                        kpts1,
                        data["T_0to1"],
                        depth_matches,
                        estimate="relpose",
                    )

                    metrics = {
                        **metrics,
                        **{"desc_" + k: v for k, v in desc_metrics.items()},
                        **{"depth_" + k: v for k, v in depth_metrics.items()},
                    }
                else:
                    desc_matches = self.val_matcher.match_by_descriptors(pred)
                    matches = self.val_matcher.match_by_homography(data, pred)
                    kpts0 = pred["keypoints0"]
                    kpts1 = pred["keypoints1"]

                    desc_metrics = get_match_metrics(
                        kpts0,
                        kpts1,
                        data["H_0to1"],
                        desc_matches,
                        data["view0"]["image_size"],
                        estimate="homography",
                    )
                    homography_metrics = get_match_metrics(
                        kpts0,
                        kpts1,
                        data["H_0to1"],
                        matches,
                        data["view0"]["image_size"],
                        estimate="homography",
                    )

                    metrics = {
                        **metrics,
                        **{"desc_" + k: v for k, v in desc_metrics.items()},
                        **{"homography_" + k: v for k, v in homography_metrics.items()},
                    }
        else:
            metrics = {}

        return losses, metrics

    def _detach_grad_filter(self, key):
        if key.startswith("keypoint_scores") or key.startswith("descriptors"):
            return True
        else:
            return False
