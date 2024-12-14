from pathlib import Path

import torch

from ...geometry.depth import simple_project, unproject
from ...geometry.epipolar import (
    T_to_F,
    asymm_epipolar_distance_all,
    relative_pose_error,
)
from ...geometry.homography import homography_corner_error
from ...robust_estimators import load_estimator
from ...settings import DATA_PATH, TRAINING_PATH
from ..base_model import BaseModel
from ..utils.misc import (
    CycleMatcher,
    classify_by_epipolar,
    classify_by_homography,
    lscore,
    pad_and_stack,
    select_on_last,
    tile,
    reproject_homography,
)
from ..utils.unet import Unet


def point_distribution_disk(logits):
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


def point_distribution(logits, budget):
    proposal_dist = torch.distributions.Categorical(logits=logits)
    proposals = proposal_dist.sample()
    proposal_logp = proposal_dist.log_prob(proposals)

    accept_logits = select_on_last(logits, proposals).squeeze(-1)

    b, tiled_h, tiled_w = accept_logits.shape

    flat_logits = accept_logits.reshape(b, -1)

    gumbel_dist = torch.distributions.Gumbel(0, 1)
    gumbel_scores = flat_logits + gumbel_dist.sample(flat_logits.shape).to(
        logits.device
    )
    if budget > flat_logits.shape[-1]:
        budget = flat_logits.shape[-1]
    topk_indices = torch.topk(gumbel_scores, budget, dim=-1).indices

    accept_samples = torch.zeros_like(flat_logits)
    accept_samples.scatter_(-1, topk_indices, 1)
    accept_samples = accept_samples.reshape(b, tiled_h, tiled_w)

    accept_mask = accept_samples == 1

    accept_logp = torch.log_softmax(accept_logits, dim=-1)
    accept_logp = accept_logp.reshape(b, tiled_h, tiled_w)
    logp = proposal_logp + accept_logp

    return proposals, accept_mask, logp


def epipolar_reward(data, pred, threshold=2.0, score_type="coarse", lm_e=0.25):
    kpts0 = pred["keypoints0"]
    kpts1 = pred["keypoints1"]

    camera0 = data["view0"]["camera"]
    camera1 = data["view1"]["camera"]

    F0_1 = T_to_F(camera0, camera1, data["T_0to1"])
    F1_0 = T_to_F(camera1, camera0, data["T_1to0"])

    dist0 = asymm_epipolar_distance_all(kpts0, kpts1, F0_1).abs() ** 2
    dist1 = asymm_epipolar_distance_all(kpts1, kpts0, F1_0).abs() ** 2

    edist_error0 = torch.min(dist0.nan_to_num(nan=float("inf")), dim=-1).values
    edist_error1 = torch.min(dist1.nan_to_num(nan=float("inf")), dim=-1).values

    score0 = lscore(edist_error0, threshold, type=score_type)
    score1 = lscore(edist_error1, threshold, type=score_type)

    return score0, score1


def depth_reward(data, pred, threshold=2.0, score_type="coarse", lm_e=0.25):
    kpts0 = pred["keypoints0"]
    kpts1 = pred["keypoints1"]

    depth0 = data["view0"]["depth"]
    depth1 = data["view1"]["depth"]

    camera0 = data["view0"]["camera"]
    camera1 = data["view1"]["camera"]

    T0 = data["view0"]["T_w2cam"]
    T1 = data["view1"]["T_w2cam"]

    kpts0_r = simple_project(unproject(kpts0, depth0, camera0, T0), camera1, T1)
    kpts1_r = simple_project(unproject(kpts1, depth1, camera1, T1), camera0, T0)

    diff0 = kpts0[:, :, None, :] - kpts1_r[:, None, :, :]
    diff1 = kpts1[:, :, None, :] - kpts0_r[:, None, :, :]

    dist0 = torch.norm(diff0, p=2, dim=-1)
    dist1 = torch.norm(diff1, p=2, dim=-1)

    reproj_error0 = torch.min(dist0.nan_to_num(nan=float("inf")), dim=-1).values
    reproj_error1 = torch.min(dist1.nan_to_num(nan=float("inf")), dim=-1).values

    escore0, escore1 = epipolar_reward(data, pred, threshold, score_type, lm_e)

    score0 = lscore(reproj_error0, threshold, type=score_type) + lm_e * escore0
    score1 = lscore(reproj_error1, threshold, type=score_type) + lm_e * escore1

    return score0, score1


def homography_reward(data, pred, threshold=2.0, score_type="coarse", lm_e=0.25):
    kpts0 = pred["keypoints0"]
    kpts1 = pred["keypoints1"]

    H_0to1 = data["H_0to1"]

    kpts0_r = reproject_homography(kpts0, H_0to1, data["view1"]["image_size"], False)
    kpts1_r = reproject_homography(kpts1, H_0to1, data["view0"]["image_size"], True)

    diff0 = kpts0[:, :, None, :] - kpts1_r[:, None, :, :]
    diff1 = kpts1[:, :, None, :] - kpts0_r[:, None, :, :]

    dist0 = torch.norm(diff0, p=2, dim=-1)
    dist1 = torch.norm(diff1, p=2, dim=-1)

    reproj_error0 = torch.min(dist0.nan_to_num(nan=float("inf")), dim=-1).values
    reproj_error1 = torch.min(dist1.nan_to_num(nan=float("inf")), dim=-1).values

    score0 = lscore(reproj_error0, threshold, type=score_type)
    score1 = lscore(reproj_error1, threshold, type=score_type)

    return score0, score1


class DISK(BaseModel):
    default_conf = {
        "window_size": 8,
        "nms_radius": 2,  # matches with disk nms radius of 5
        "max_num_keypoints": None,
        "sample_budget": 4096,
        "force_num_keypoints": False,
        "pad_if_not_divisible": True,
        "weights": None,
        "reward": "depth",
        "pad_edges": 4,
        "eval_sampling": "nms",
        "arch": {
            "kernel_size": 5,
            "gate": "PReLU",
            "norm": "InstanceNorm2d",
            "down": [16, 32, 64, 64, 64],
            "up": [64, 64, 64, 1],
            "upsample": "TrivialUpsample",
            "downsample": "TrivialDownsample",
            "down_block": "ThinDownBlock",  # second option is DownBlock
            "up_block": "ThinUpBlock",  # second option is UpBlock
            # "dropout": False, not used yet
            "bias": True,
            "padding": True,
        },
        "loss": {
            "score_type": "coarse",
            "reward_threshold": 1.5,
            "lm_e": 0.1,
        },
        "estimator": {"name": "degensac", "ransac_th": 1.0},
    }

    required_data_keys = []

    def _init(self, conf):
        self.set_initialized()

        self.unet = Unet(
            in_features=3,
            conf=self.conf,
        )

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

        self.val_matcher = CycleMatcher(self.conf.loss.reward_threshold)

    def _sample(self, heatmaps, budget_override=None, nms=False):
        v = self.conf.window_size
        device = heatmaps.device
        b, _, h, w = heatmaps.shape

        assert h % v == 0
        assert w % v == 0

        tiled = tile(heatmaps, self.conf.window_size).squeeze(1)

        budget = (
            budget_override if budget_override is not None else self.conf.sample_budget
        )

        if budget is None:
            proposals, accept_mask, logp = point_distribution_disk(tiled)
        else:
            proposals, accept_mask, logp = point_distribution(tiled, budget)

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

        if nms:
            updated_heatmaps = torch.full_like(heatmaps, -float("inf"))

            for i in range(b):
                updated_heatmaps[i, 0, keypoints[i][:, 1], keypoints[i][:, 0]] = scores[
                    i
                ]

            keypoints, scores = self._nms(updated_heatmaps, -float("inf"))

        return keypoints, scores

    def _nms(self, heatmaps, detection_threshold=None):
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

        if detection_threshold is not None:
            nms = nms & (heatmaps > detection_threshold)

        keypoints = []
        scores = []

        for i in range(heatmaps.shape[0]):
            keypoints.append(torch.flip(nms[i].nonzero(as_tuple=False), (1,)))
            scores.append(heatmaps[i][nms[i]])

        return keypoints, scores

    def get_heatmap(self, image0=None, image1=None):
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
            heatmap0, heatmap1 = self.unet(input1=image0, input2=image1)

            return heatmap0, heatmap1
        elif image0 is not None:
            heatmap = self.unet(input1=image0)
            return heatmap
        else:
            heatmap = self.unet(input1=image1)
            return heatmap

    def _forward(self, data):
        def process_heatmap(heatmap, image_size):
            minval = heatmap.detach().min()
            heatmap[:, :, : self.conf.pad_edges, :] = minval
            heatmap[:, :, :, : self.conf.pad_edges] = minval
            for i in range(heatmap.shape[0]):
                h, w = image_size[i].long()
                heatmap[i, :, h.item() - self.conf.pad_edges :, :] = minval
                heatmap[i, :, :, w.item() - self.conf.pad_edges :] = minval

            disable_filter = False
            if self.training:
                # Use sampling during training
                points, logps = self._sample(heatmap, nms=False)
                disable_filter = True
            else:
                if self.conf.eval_sampling == "nms":
                    # Use NMS during evaluation
                    points, logps = self._nms(heatmap, 0.0)
                    disable_filter = False
                elif self.conf.eval_sampling == "budget":
                    points, logps = self._sample(
                        heatmap, self.conf.max_num_keypoints, nms=True
                    )
                    disable_filter = True
                elif self.conf.eval_sampling == "disk":
                    points, logps = self._sample(heatmap, budget_override=-1, nms=False)
                    disable_filter = False

            keypoints = []
            scores = []

            for i, (point, logp) in enumerate(zip(points, logps)):
                if not disable_filter and self.conf.max_num_keypoints is not None:
                    n = min(self.conf.max_num_keypoints + 1, logp.numel())
                    minus_threshold, _ = torch.kthvalue(-logp, n)
                    mask = logp > -minus_threshold

                    point = point[mask]
                    logp = logp[mask]

                # filter points outside of image_shape
                image_shape = image_size[i].flip(dims=(0,))
                padding = 0 if self.conf.pad_edges is None else self.conf.pad_edges
                vis = torch.all(
                    (point > padding) & (point < image_shape - padding),
                    dim=-1,
                )
                point = point[vis]
                logp = logp[vis]

                x, y = point.T

                keypoints.append(point)
                scores.append(logp)

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
            else:
                keypoints = torch.stack(keypoints, 0)
                scores = torch.stack(scores, 0)

            return {
                "keypoints": keypoints.float(),
                "heatmap": heatmap,
                "keypoint_scores": scores.float(),
            }

        if "view0" in data:
            if "view1" in data:
                heatmap0, heatmap1 = self.get_heatmap(
                    data["view0"]["image"], data["view1"]["image"]
                )
                pred = {}
                pred0 = process_heatmap(heatmap0, data["view0"]["image_size"])
                pred1 = process_heatmap(heatmap1, data["view1"]["image_size"])
                pred = {f"{k}0": v for k, v in pred0.items()}
                pred = {**pred, **{f"{k}1": v for k, v in pred1.items()}}
                return pred
            else:
                heatmap = self.get_heatmap(data["view0"]["image"])
            return process_heatmap(heatmap, data["view0"]["image_size"])
        else:
            heatmap = self.get_heatmap(data["image"])
            return process_heatmap(heatmap, data["image_size"])

    def _pre_loss_callback(self, seed, epoch):
        pass

    def loss(self, pred, data):
        if self.conf.reward == "depth":
            elementwise_reward0, elementwise_reward1 = depth_reward(
                data,
                pred,
                threshold=self.conf.loss.reward_threshold,
                score_type=self.conf.loss.score_type,
                lm_e=self.conf.loss.lm_e,
            )
        elif self.conf.reward == "epipolar":
            elementwise_reward0, elementwise_reward1 = epipolar_reward(
                data,
                pred,
                threshold=self.conf.loss.reward_threshold,
                score_type=self.conf.loss.score_type,
            )
        elif self.conf.reward == "homography":
            elementwise_reward0, elementwise_reward1 = homography_reward(
                data,
                pred,
                threshold=self.conf.loss.reward_threshold,
                score_type=self.conf.loss.score_type,
            )
        else:
            raise ValueError(f"Unknown reward type {self.conf.reward}")

        logp0 = pred["keypoint_scores0"]
        logp1 = pred["keypoint_scores1"]

        reinforce = (elementwise_reward0 * logp0).sum(dim=-1) + (
            elementwise_reward1 * logp1
        ).sum(dim=-1)

        loss = -reinforce

        losses = {
            "total": loss,
            "reinforce": reinforce,
            "n_kpts": (
                torch.count_nonzero(logp0, dim=-1) + torch.count_nonzero(logp1, dim=-1)
            ).float(),
        }
        del (logp0, logp1, reinforce, loss)

        metrics = {}
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
                    "n_kpts": torch.count_nonzero(pred["keypoint_scores0"], dim=-1)
                    + torch.count_nonzero(pred["keypoint_scores1"], dim=-1),
                }

                if "depth" in data["view0"]:
                    depth_matches = self.val_matcher.match_by_depth(data, pred)
                    kpts0 = pred["keypoints0"]
                    kpts1 = pred["keypoints1"]

                    depth_metrics = get_match_metrics(
                        kpts0,
                        kpts1,
                        data["T_0to1"],
                        depth_matches,
                        estimate="relpose",
                    )

                    metrics = {
                        **metrics,
                        **{"depth_" + k: v for k, v in depth_metrics.items()},
                    }
                else:
                    matches = self.val_matcher.match_by_homography(data, pred)
                    kpts0 = pred["keypoints0"]
                    kpts1 = pred["keypoints1"]

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
                        **{"homography_" + k: v for k, v in homography_metrics.items()},
                    }
        else:
            metrics = {}

        return losses, metrics

    def _detach_grad_filter(self, key):
        if key.startswith("keypoint_scores"):
            return True
        else:
            return False
