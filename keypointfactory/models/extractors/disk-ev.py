from pathlib import Path

import torch

from ...geometry.epipolar import T_to_F, asymm_epipolar_distance_all
from ...settings import DATA_PATH, TRAINING_PATH
from ..base_model import BaseModel
from ..utils.unet import Unet
from ..utils.misc import (
    pad_and_stack,
    tile,
    select_on_last,
    lscore,
    and_mult,
    reproject_homography,
    unproject,
    project,
)


def point_distribution(logits, budget):
    proposal_dist = torch.distributions.Categorical(logits=logits)
    proposals = proposal_dist.sample()
    proposal_logp = proposal_dist.log_prob(proposals)

    accept_logits = select_on_last(logits, proposals).squeeze(-1)

    if budget != -1:
        b, tiled_h, tiled_w = accept_logits.shape
        
        flat_logits = accept_logits.reshape(b, -1)

        gumbel_dist = torch.distributions.Gumbel(0, 1)
        gumbel_scores = flat_logits + gumbel_dist.sample(flat_logits.shape).to(logits.device)
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
        kpts_r = project(unproject(kpts_o, depth_o, cam_o, T_o), cam, T)

        diff = kpts_r[:, None, :, :] - kpts[:, :, None, :]

        dist = torch.norm(diff, p=2, dim=-1)

        reproj_error = torch.min(dist.nan_to_num(nan=float("inf")), dim=-1).values

        return lscore(reproj_error, threshold, score_type)

    error1_0 = get_score(kpts0, kpts1, T0, T1, camera0, camera1, depth1)
    error2_0 = get_score(kpts0, kpts2, T0, T2, camera0, camera2, depth2)
    error0_1 = get_score(kpts1, kpts0, T1, T0, camera1, camera0, depth0)
    error2_1 = get_score(kpts1, kpts2, T1, T2, camera1, camera2, depth2)
    error0_2 = get_score(kpts2, kpts0, T2, T0, camera2, camera0, depth0)
    error1_2 = get_score(kpts2, kpts1, T2, T1, camera2, camera1, depth1)

    reward0 = and_mult(error1_0, error2_0)
    reward1 = and_mult(error0_1, error2_1)
    reward2 = and_mult(error0_2, error1_2)

    # substitute epipolar reward for when depth is not available. Multiple by lm_e to make it less important

    return reward0, reward1, reward2


def epipolar_reward(data, pred, threshold=3, score_type="linear", lm_e=0.25):
    kpts0, kpts1, kpts2 = pred["keypoints0"], pred["keypoints1"], pred["keypoints2"]

    F_0to1 = T_to_F(data["view0"]["camera"], data["view1"]["camera"], pred["T_0to1"])
    F_0to2 = T_to_F(data["view0"]["camera"], data["view2"]["camera"], pred["T_0to2"])
    F_1to2 = T_to_F(data["view1"]["camera"], data["view2"]["camera"], pred["T_1to2"])

    def get_score(kpts, kpts_o, F, inverse):
        if inverse:
            F = F.inverse()

        dist = asymm_epipolar_distance_all(kpts_o, kpts, F).abs()

        reproj_error = torch.min(dist.nan_to_num(nan=float("inf")), dim=-1).values

        return lscore(reproj_error, threshold, score_type)

    error1_0 = get_score(kpts0, kpts1, F_0to1, True)
    error2_0 = get_score(kpts0, kpts2, F_0to2, True)
    error0_1 = get_score(kpts1, kpts0, F_0to1, False)
    error2_1 = get_score(kpts1, kpts2, F_1to2, True)
    error0_2 = get_score(kpts2, kpts0, F_0to2, False)
    error1_2 = get_score(kpts2, kpts1, F_1to2, False)

    reward0 = and_mult(error1_0, error2_0)
    reward1 = and_mult(error0_1, error2_1)
    reward2 = and_mult(error0_2, error1_2)

    return reward0, reward1, reward2


def homography_reward(data, pred, threshold=3, score_type="linear", lm_e=0.25):
    kpts0, kpts1, kpts2 = pred["keypoints0"], pred["keypoints1"], pred["keypoints2"]

    H_0to1 = pred["H_0to1"]
    H_0to2 = pred["H_0to2"]
    H_1to2 = pred["H_1to2"]

    def get_score(kpts, kpts_o, H, type, img_bounds, inverse):
        kpts_r = reproject_homography(kpts_o, H, *img_bounds, inverse)
        diff = kpts_r[:, None, :, :] - kpts[:, :, None, :]

        dist = torch.norm(diff, p=2, dim=-1)

        reproj_error = torch.min(dist.nan_to_num(nan=float("inf")), dim=-1).values

        return lscore(reproj_error, threshold, score_type)

    error1_0 = get_score(kpts0, kpts1, H_0to1, data["view0"]["image"].shape[2:], True)
    error2_0 = get_score(kpts0, kpts2, H_0to2, data["view0"]["image"].shape[2:], True)
    error0_1 = get_score(kpts1, kpts0, H_0to1, data["view1"]["image"].shape[2:], False)
    error2_1 = get_score(kpts1, kpts2, H_1to2, data["view1"]["image"].shape[2:], True)
    error0_2 = get_score(kpts2, kpts0, H_0to2, data["view2"]["image"].shape[2:], False)
    error1_2 = get_score(kpts2, kpts1, H_1to2, data["view2"]["image"].shape[2:], False)

    reward0 = and_mult(error1_0, error2_0)
    reward1 = and_mult(error0_1, error2_1)
    reward2 = and_mult(error0_2, error1_2)

    return reward0, reward1, reward2


class DISK_EV(BaseModel):
    default_conf = {
        "window_size": 8,
        "nms_radius": 2,  # matches with disk nms radius of 5
        "max_num_keypoints": None,
        "force_num_keypoints": False,
        "pad_if_not_divisible": True,
        "detection_threshold": 0.005,
        "weights": None,
        "reward": "depth",
        "arch": {
            "kernel_size": 5,
            "gate": "PReLU",
            "norm": "InstanceNorm2d",
            "upsample": "TrivialUpsample",
            "downsample": "TrivialDownsample",
            "down_block": "ThinDownBlock",
            "up_block": "ThinUpBlock",
            # "dropout": False, not used yet
            "bias": True,
            "padding": True,
            "train_invT": False,
        },
        "loss": {
            "reward_threshold": 1.5,
            "score_type": "linear",  # coarse, fine, linear, or discrete
            "lambda_e": 0.25,
        },
        "estimator": {"name": "degensac", "ransac_th": 1.0},
    }

    required_data_keys = ["image"]

    def _init(self, conf):
        self.set_initialized()

        self.unet = Unet(in_features=3, down=[16, 32, 64], up=[64, 64, 1])

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

    def _forward(self, data):
        images = data["image"]
        if self.conf.pad_if_not_divisible:
            h, w = images.shape[2:]
            pd_h = 16 - h % 16 if h % 16 > 0 else 0
            pd_w = 16 - w % 16 if w % 16 > 0 else 0
            images = torch.nn.functional.pad(images, (0, pd_w, 0, pd_h), value=0.0)

        heatmap = self.unet(images)

        if self.training:
            # Use sampling during training
            points, logps = self._sample(heatmap)
        else:
            # Use NMS during evaluation
            points, logps = self._nms(heatmap)

        keypoints = []
        scores = []

        for i, (point, logp) in enumerate(zip(points, logps)):
            if self.conf.max_num_keypoints is not None:
                n = min(self.conf.max_num_keypoints + 1, logp.numel())
                minus_threshold, _ = torch.kthvalue(-logp, n)
                mask = logp > -minus_threshold

                point = point[mask]
                logp = logp[mask]

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

    def _pre_loss_callback(self, seed, epoch):
        pass

    def loss(self, data, pred):
        if self.conf.reward == "depth":
            element_wise_rewards = depth_reward(
                data,
                pred,
                threshold=self.conf.loss.reward_threshold,
                score_type=self.conf.loss.score_type,
                lm_e=self.conf.loss.lambda_e,
            )
        elif self.conf.reward == "epipolar":
            element_wise_rewards = epipolar_reward(
                data,
                pred,
                threshold=self.conf.loss.reward_threshold,
                score_type=self.conf.loss.score_type,
                lm_e=self.conf.loss.lambda_e,
            )
        elif self.conf.reward == "homography":
            element_wise_rewards = homography_reward(
                data,
                pred,
                threshold=self.conf.loss.reward_threshold,
                score_type=self.conf.loss.score_type,
                lm_e=self.conf.loss.lambda_e,
            )
        else:
            raise ValueError(f"Unknown reward type {self.conf.reward}")

        logp0, logp1, logp2 = (
            pred["keypoint_scores0"],
            pred["keypoint_scores1"],
            pred["keypoint_scores2"],
        )

        reward0, reward1, reward2 = element_wise_rewards

        reinforce = (
            (reward0 * logp0).sum(dim=-1)
            + (reward1 * logp1).sum(dim=-1)
            + (reward2 * logp2).sum(dim=-1)
        )

        loss = -reinforce

        losses = {
            "total": loss,
            "reinforce": reinforce,
            "n_kpts": (
                torch.count_nonzero(logp0, dim=-1)
                + torch.count_nonzero(logp1)
                + torch.count_nonzero(logp2)
            ).float(),
        }

        del (
            logp0,
            logp1,
            logp2,
            element_wise_rewards,
            reward0,
            reward1,
            reward2,
            reinforce,
            loss,
        )

        metrics = {}
        if not self.training:
            # Calculate validation metrics
            pass

        return losses, metrics
