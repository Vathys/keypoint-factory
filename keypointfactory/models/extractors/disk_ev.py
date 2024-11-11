from pathlib import Path

import torch
from omegaconf import OmegaConf

from ...geometry.homography import homography_corner_error
from ...geometry.epipolar import (
    T_to_F,
    asymm_epipolar_distance_all,
    relative_pose_error,
)
from ...geometry.depth import simple_project, unproject
from ...robust_estimators import load_estimator
from ...settings import DATA_PATH, TRAINING_PATH
from ..base_model import BaseModel
from ..utils.unet import Unet
from ..utils.misc import (
    distance_matrix,
    pad_and_stack,
    tile,
    select_on_last,
    lscore,
    reproject_homography,
    CycleMatcher,
    classify_by_epipolar,
    classify_by_homography,
)
from ...utils.misc import get_twoview
from ... import logger


def point_distribution(logits, budget=-1):
    proposal_dist = torch.distributions.Categorical(logits=logits)
    proposals = proposal_dist.sample()
    proposal_logp = proposal_dist.log_prob(proposals)

    accept_logits = select_on_last(logits, proposals).squeeze(-1)

    if budget != -1:
        b, tiled_h, tiled_w = accept_logits.shape

        flat_logits = accept_logits.reshape(b, -1)

        gumbel_dist = torch.distributions.Gumbel(0, 1)
        gumbel_scores = flat_logits + gumbel_dist.sample(flat_logits.shape).to(
            logits.device
        )
        topk_indices = torch.topk(gumbel_scores, budget, dim=-1).indices

        accept_samples = torch.zeros_like(flat_logits)
        accept_samples.scatter_(-1, topk_indices, 1)
        accept_samples = accept_samples.reshape(b, tiled_h, tiled_w)

        accept_mask = accept_samples == 1

        accept_logp = torch.log_softmax(accept_logits, dim=-1)
        accept_logp = accept_logp.reshape(b, tiled_h, tiled_w)
        logp = proposal_logp + accept_logp
    else:
        accept_mask = torch.ones_like(accept_logits, dtype=torch.bool)
        logp = proposal_logp

    return proposals, accept_mask, logp


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


def depth_reward(data, pred, threshold=3, score_type="linear", lm_e=0.25):
    kpts0, kpts1, kpts2 = pred["keypoints0"], pred["keypoints1"], pred["keypoints2"]

    depth0, depth1, depth2 = (
        data["view0"]["depth"],
        data["view1"]["depth"],
        data["view2"]["depth"],
    )
    camera0, camera1, camera2 = (
        data["view0"]["camera"],
        data["view1"]["camera"],
        data["view2"]["camera"],
    )
    T0, T1, T2 = (
        data["view0"]["T_w2cam"],
        data["view1"]["T_w2cam"],
        data["view2"]["T_w2cam"],
    )

    def get_score(kpts, kpts_o, T, T_o, cam, cam_o, depth_o):
        kpts_r = simple_project(unproject(kpts_o, depth_o, cam_o, T_o), cam, T)

        diff = kpts[:, :, None, :] - kpts_r[:, None, :, :]

        dist = torch.norm(diff, p=2, dim=-1)

        reproj_error = torch.min(dist.nan_to_num(nan=float("inf")), dim=-1).values

        return lscore(reproj_error, threshold, score_type, rescale=True)

    error1_0 = get_score(kpts0, kpts1, T0, T1, camera0, camera1, depth1)
    error2_0 = get_score(kpts0, kpts2, T0, T2, camera0, camera2, depth2)
    error0_1 = get_score(kpts1, kpts0, T1, T0, camera1, camera0, depth0)
    error2_1 = get_score(kpts1, kpts2, T1, T2, camera1, camera2, depth2)
    error0_2 = get_score(kpts2, kpts0, T2, T0, camera2, camera0, depth0)
    error1_2 = get_score(kpts2, kpts1, T2, T1, camera2, camera1, depth1)

    ereward0, ereward1, ereward2 = epipolar_reward(data, pred, threshold, score_type)

    # reward0 = and_mult(error1_0, error2_0, rescale=True) + lm_e * ereward0
    # reward1 = and_mult(error0_1, error2_1, rescale=True) + lm_e * ereward1
    # reward2 = and_mult(error0_2, error1_2, rescale=True) + lm_e * ereward2

    reward0 = (error1_0 + error2_0) * 0.5 + lm_e * ereward0
    reward1 = (error0_1 + error2_1) * 0.5 + lm_e * ereward1
    reward2 = (error0_2 + error1_2) * 0.5 + lm_e * ereward2

    return {"reward0": reward0, "reward1": reward1, "reward2": reward2}


def epipolar_reward(data, pred, threshold=3, score_type="linear", lm_e=0.25):
    kpts0, kpts1, kpts2 = pred["keypoints0"], pred["keypoints1"], pred["keypoints2"]

    camera0, camera1, camera2 = (
        data["view0"]["camera"],
        data["view1"]["camera"],
        data["view2"]["camera"],
    )

    def get_score(kpts, kpts_o, camera, camera_o, T):
        F = T_to_F(camera, camera_o, T)

        dist = asymm_epipolar_distance_all(kpts, kpts_o, F).abs()

        reproj_error = torch.min(dist.nan_to_num(nan=float("inf")), dim=-1).values

        return lscore(reproj_error, threshold, score_type, rescale=True)

    error1_0 = get_score(kpts0, kpts1, camera0, camera1, data["T_0to1"])
    error2_0 = get_score(kpts0, kpts2, camera0, camera2, data["T_0to2"])
    error0_1 = get_score(kpts1, kpts0, camera1, camera0, data["T_1to0"])
    error2_1 = get_score(kpts1, kpts2, camera1, camera2, data["T_1to2"])
    error0_2 = get_score(kpts2, kpts0, camera2, camera0, data["T_2to0"])
    error1_2 = get_score(kpts2, kpts1, camera2, camera1, data["T_2to1"])

    # reward0 = and_mult(error1_0, error2_0, rescale=True)
    # reward1 = and_mult(error0_1, error2_1, rescale=True)
    # reward2 = and_mult(error0_2, error1_2, rescale=True)

    reward0 = (error1_0 + error2_0) * 0.5
    reward1 = (error0_1 + error2_1) * 0.5
    reward2 = (error0_2 + error1_2) * 0.5

    return {"reward0": reward0, "reward1": reward1, "reward2": reward2}


def homography_reward(data, pred, threshold=3, score_type="linear", lm_e=0.25):
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

    close0 = torch.norm(diff0, p=2, dim=-1)
    close1 = torch.norm(diff1, p=2, dim=-1)

    return (
        lscore(close0, threshold, score_type, rescale=True)
        + lscore(close1, threshold, score_type, rescale=True)
    ) / 2


class DISK_EV(BaseModel):
    default_conf = {
        "window_size": 8,
        "nms_radius": 2,  # matches with disk nms radius of 5
        "max_num_keypoints": None,
        "sample_budget": 4096,
        "force_num_keypoints": False,
        "pad_if_not_divisible": True,
        "detection_threshold": 0.005,
        "weights": None,
        "reward": "depth",
        "pad_edges": 4,
        "desc_dim": 128,
        "arch": {
            "kernel_size": 5,
            "gate": "PReLU",
            "norm": "InstanceNorm2d",
            "down": [16, 32, 64, 64, 64],
            "up": [64, 64, 64],
            "upsample": "TrivialUpsample",
            "downsample": "TrivialDownsample",
            "down_block": "ThinDownBlock",
            "up_block": "ThinUpBlock",
            # "dropout": False, not used yet
            "bias": True,
            "padding": True,
        },
        "train_invT": True,
        "loss": {
            "reward_threshold": 1.5,
            "score_type": "linear",  # coarse, fine, linear, or discrete
            "lambda_e": 0.25,
            "lambda_kp": -0.001,
        },
        "estimator": {"name": "degensac", "ransac_th": 1.0},
    }

    required_data_keys = []

    def _init(self, conf):
        self.set_initialized()

        self.conf = OmegaConf.merge(
            self.conf,
            OmegaConf.create(
                {
                    "arch": {
                        "down": self.conf.arch.down,
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
        self.train_matcher.requires_grad_(self.conf.train_invT)

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
            model_sd = {}
            for k, v in state_dict["model"].items():
                model_sd[k.replace("extractor.", "")] = v
            missing_keys, unexpected_keys = self.load_state_dict(model_sd, strict=False)

            print(missing_keys)
            print(unexpected_keys)

        self.val_matcher = CycleMatcher(self.conf.loss.reward_threshold)

        self.lm_kp = self.conf.loss.lambda_kp

    def _sample(self, heatmaps):
        v = self.conf.window_size
        device = heatmaps.device
        b, _, h, w = heatmaps.shape

        assert h % v == 0
        assert w % v == 0

        tiled = tile(heatmaps, self.conf.window_size).squeeze(1)

        proposals, accept_mask, logp = point_distribution(
            tiled, self.conf.sample_budget
        )

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

    def get_heatmap_and_descriptors(self, image0=None, image1=None):
        assert image0 is not None or image1 is not None
        if self.conf.pad_if_not_divisible:
            if image0 is not None:
                h, w = image0.shape[2:]
                pd_h = 16 - h % 16 if h % 16 > 0 else 0
                pd_w = 16 - w % 16 if w % 16 > 0 else 0
                image0 = torch.nn.functional.pad(image0, (0, pd_w, 0, pd_h), value=0.0)

            if image1 is not None:
                h, w = image1.shape[2:]
                pd_h = 16 - h % 16 if h % 16 > 0 else 0
                pd_w = 16 - w % 16 if w % 16 > 0 else 0
                image1 = torch.nn.functional.pad(image1, (0, pd_w, 0, pd_h), value=0.0)

        if image0 is not None and image1 is not None:
            output0, output1 = self.unet(input1=image0, input2=image1)
            desc0 = output0[:, : self.conf.desc_dim]
            heatmap0 = output0[:, self.conf.desc_dim :]
            desc1 = output1[:, : self.conf.desc_dim]
            heatmap1 = output1[:, self.conf.desc_dim :]

            return heatmap0, desc0, heatmap1, desc1
        elif image0 is not None:
            output = self.unet(input1=image0)
            desc = output[:, : self.conf.desc_dim]
            heatmap = output[:, self.conf.desc_dim :]
            return heatmap, desc
        else:
            output = self.unet(input1=image1)
            desc = output[:, : self.conf.desc_dim]
            heatmap = output[:, self.conf.desc_dim :]
            return heatmap, desc

    def _forward(self, data):
        def process_heatmap(heatmap, descs, image_size):
            if self.training:
                # Use sampling during training
                points, logps = self._sample(heatmap)
            else:
                # Use NMS during evaluation
                points, logps = self._nms(heatmap)

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

                if self.conf.pad_edges > 0:
                    image_shape = image_size[i]
                    vis = torch.all(
                        (point > self.conf.pad_edges)
                        & (point < image_shape - self.conf.pad_edges),
                        dim=-1,
                    )
                    point = point[vis]
                    logp = logp[vis]

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
                descriptors = pad_and_stack(
                    descriptors, target_length, -2, mode="zeros"
                )
            else:
                keypoints = torch.stack(keypoints, 0)
                scores = torch.stack(scores, 0)
                descriptors = torch.stack(descriptors, 0)

            return {
                "keypoints": keypoints.float(),
                "heatmap": heatmap,
                "keypoint_scores": scores.float(),
                "descriptors": descriptors.float(),
            }

        if "view0" in data:
            if "view1" in data:
                heatmap0, desc0, heatmap1, desc1 = self.get_heatmap_and_descriptors(
                    data["view0"]["image"], data["view1"]["image"]
                )
                pred = {}
                pred0 = process_heatmap(heatmap0, desc0, data["view0"]["image_size"])
                pred1 = process_heatmap(heatmap1, desc1, data["view1"]["image_size"])
                pred = {f"{k}0": v for k, v in pred0.items()}
                pred = {**pred, **{f"{k}1": v for k, v in pred1.items()}}
                return pred
            else:
                heatmap, desc = self.get_heatmap_and_descriptors(data["view0"]["image"])
            return process_heatmap(heatmap, desc, data["view0"]["image_size"])
        else:
            heatmap, desc = self.get_heatmap_and_descriptors(data["image"])
            return process_heatmap(heatmap, desc, data["image_size"])

    def _pre_loss_callback(self, seed, epoch):
        if epoch == 0:
            ramp = 0
        elif epoch == 1:
            ramp = 0.1
        else:
            ramp = min(1.0, 0.1 + 0.2 * epoch)

        self.lm_kp = self.conf.loss.lambda_kp * ramp

    def _get_validation_metrics(self, data, pred, rewards):
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

            def get_match_metrics(
                kpts0,
                kpts1,
                elementwise_reward0,
                elementwise_reward1,
                M_gt,
                matches,
                image_size=None,
                estimate="relpose",
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
                        data,
                        {"keypoints0": kpts0, "keypoints1": kpts1},
                        threshold=self.conf.loss.reward_threshold,
                    )
                elif estimate == "homography":
                    good = classify_by_homography(
                        data,
                        {"keypoints0": kpts0, "keypoints1": kpts1},
                        threshold=self.conf.loss.reward_threshold,
                    )
                good = good.diagonal(dim1=-2, dim2=-1)
                bad = ~good

                results["n_good"] = good.to(torch.int64).sum(dim=1)
                results["n_bad"] = bad.to(torch.int64).sum(dim=1)
                results["prec"] = results["n_good"] / (results["n_pairs"] + 1)

                results["reward"] = elementwise_reward0.sum(
                    dim=1
                ) + elementwise_reward1.sum(dim=1)

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
                                    torch.tensor([float("inf")], device=kpts0.device),
                                ]
                            )
                        elif estimate == "homography":
                            results["H_error"] = torch.cat(
                                [
                                    results["H_error"],
                                    torch.tensor([float("inf")], device=kpts0.device),
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
                            t_error, r_error = relative_pose_error(M_gt[b], M.R, M.t)

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
                "n_kpts": torch.count_nonzero(pred["keypoint_scores0"], dim=-1)
                + torch.count_nonzero(pred["keypoint_scores1"], dim=-1),
            }

            if "depth" in data["view0"]:
                desc_matches = self.val_matcher.match_by_descriptors(pred)
                depth_matches = self.val_matcher.match_by_depth(data, pred)
                kpts0 = pred["keypoints0"]
                kpts1 = pred["keypoints1"]
                elementwise_reward0 = rewards["reward0"]
                elementwise_reward1 = rewards["reward1"]

                desc_metrics = get_match_metrics(
                    kpts0,
                    kpts1,
                    elementwise_reward0,
                    elementwise_reward1,
                    data["T_0to1"],
                    desc_matches,
                    estimate="relpose",
                )
                depth_metrics = get_match_metrics(
                    kpts0,
                    kpts1,
                    elementwise_reward0,
                    elementwise_reward1,
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
                matches = self.val_matcher.match_by_homography(data, pred)
                kpts0 = pred["keypoints0"]
                kpts1 = pred["keypoints1"]
                elementwise_reward0 = rewards["reward0"]
                elementwise_reward1 = rewards["reward1"]

                homography_metrics = get_match_metrics(
                    kpts0,
                    kpts1,
                    elementwise_reward0,
                    elementwise_reward1,
                    data["H_0to1"],
                    matches,
                    data["view0"]["image_size"],
                    estimate="homography",
                )

                metrics = {
                    **metrics,
                    **{"homography_" + k: v for k, v in homography_metrics.items()},
                }

        return metrics

    def loss(self, pred, data):
        if self.conf.reward == "depth":
            element_wise_reward = depth_reward(
                data,
                pred,
                threshold=self.conf.loss.reward_threshold,
                score_type=self.conf.loss.score_type,
                lm_e=self.conf.loss.lambda_e,
            )
        elif self.conf.reward == "epipolar":
            element_wise_reward = epipolar_reward(
                data,
                pred,
                threshold=self.conf.loss.reward_threshold,
                score_type=self.conf.loss.score_type,
                lm_e=self.conf.loss.lambda_e,
            )
        elif self.conf.reward == "homography":
            element_wise_reward = homography_reward(
                data,
                pred,
                threshold=self.conf.loss.reward_threshold,
                score_type=self.conf.loss.score_type,
                lm_e=self.conf.loss.lambda_e,
            )
        else:
            raise ValueError(f"Unknown reward type {self.conf.reward}")

        match_dist = self.train_matcher.match_pair(pred)

        with torch.no_grad():
            sample_p = match_dist.dense_p()

        sample_logp = match_dist.dense_logp()

        logp0, logp1 = (
            pred["keypoint_scores0"],
            pred["keypoint_scores1"],
        )

        kpts_logp = logp0[:, :, None] + logp1[:, None, :]

        sample_lp_flat = logp0.sum(dim=1) + logp1.sum(dim=1)

        sample_plogp = sample_p * (sample_logp + kpts_logp)

        reinforce = (element_wise_reward * sample_plogp).sum(dim=(-1, -2))

        kp_penalty = self.lm_kp * sample_lp_flat

        loss = -reinforce - kp_penalty

        losses = {
            "total": loss,
            "reinforce": reinforce,
            "kp_penalty": kp_penalty,
            "sample_lp": sample_lp_flat,
            "lm_kp": torch.tensor([self.lm_kp] * loss.shape[0], dtype=torch.float64),
            "n_kpts": torch.tensor(
                [logp0.shape[1]] * loss.shape[0],
                device=logp0.device,
                dtype=torch.float64,
            ),
            "inverse_T": torch.tensor(
                [self.train_matcher.inverse_T] * logp0.shape[0],
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

                    # results["reward"] = (
                    #     self.lm_tp * results["n_good"]
                    #     + self.lm_fp * results["n_bad"]
                    #     + self.lm_kp * n_kpts
                    # )

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
