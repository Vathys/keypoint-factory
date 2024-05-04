import torch

from ..base_model import BaseModel
from ..utils.blocks import get_module
from ..utils.misc import pad_and_stack, distance_matrix
from ...geometry.epipolar import T_to_F, asymm_epipolar_distance_all
from ...geometry.depth import sample_depth, project


def point_distribution(logits):
    proposal_dist = torch.distributions.Categorical(logits=logits)
    proposals = proposal_dist.sample()
    proposal_logp = proposal_dist.log_prob(proposals)

    accept_logits = torch.gather(logits, -1, proposals.unsqueeze(-1)).squeeze(-1)

    accept_dist = torch.distributions.Bernoulli(logits=accept_logits)
    accept_samples = accept_dist.sample()
    accept_logp = accept_dist.log_prob(accept_samples)
    accept_mask = accept_samples == 1.0

    logp = proposal_logp + accept_logp

    return proposals, accept_mask, logp


def nms(self, heatmaps, nms_radius=2):
    heatmaps = heatmaps.squeeze(1)

    ixs = torch.nn.functional.max_pool2d(
        heatmaps,
        kernel_size=2 * nms_radius + 1,
        stride=1,
        padding=nms_radius,
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
        )
        affinity = -inverse_T * distances

        self._cat_I = torch.distributions.Categorical(logits=affinity)
        self._cat_T = torch.distributions.Categorical(logits=affinity.permute(0, 2, 1))

        self._dense_logp = None
        self._dense_p = None

    def dense_p(self):
        if self._dense_p is None:
            self._dense_p = self._cat_I.probs * self._cat_T.probs.permute(0, 2, 1)

        return self._dense_p

    def dense_logp(self):
        if self._dense_logp is None:
            self._dense_logp = self._cat_I.logits + self._cat_T.logits.permute(0, 2, 1)

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


def classify_by_epipolar(data, pred, threshold=2.0):
    F_0_1 = T_to_F(data["view0"]["camera"], data["view1"]["camera"], data["T_0to1"])
    F_1_0 = T_to_F(data["view1"]["camera"], data["view0"]["camera"], data["T_1to0"])

    kpts0 = pred["keypoints0"]
    kpts1 = pred["keypoints1"]

    epi_0_1 = asymm_epipolar_distance_all(kpts0, kpts1, F_0_1).abs()
    epi_1_0 = asymm_epipolar_distance_all(kpts1, kpts0, F_1_0).abs()

    return (epi_0_1 < threshold).permute(0, 2, 1) & (epi_1_0 < threshold)


def classify_by_depth(data, pred, threshold=2.0):
    kpts0 = pred["keypoints0"]  # [B, N, 2]
    kpts1 = pred["keypoints1"]  # [B, M, 2]

    depth0 = data["view0"]["depth"]
    depth1 = data["view1"]["depth"]
    cam0 = data["view0"]["camera"]
    cam1 = data["view1"]["camera"]
    T0_1 = data["T_0to1"]
    T1_0 = data["T_1to0"]

    interp0, valid0 = sample_depth(kpts0, depth0)
    interp1, valid1 = sample_depth(kpts1, depth1)

    kpts0_1, _ = project(kpts0, interp0, None, cam0, cam1, T0_1, valid0)  # [B, M, 2]
    kpts1_0, _ = project(kpts1, interp1, None, cam1, cam0, T1_0, valid1)  # [B, N, 2]

    diff0 = kpts1_0[:, None, :, :] - kpts0[:, :, None, :]
    diff1 = kpts0_1[:, :, None, :] - kpts1[:, None, :, :]

    close0 = torch.norm(diff0, p=2, dim=-1) < threshold
    close1 = torch.norm(diff1, p=2, dim=-1) < threshold

    return close0 & close1


def epipolar_reward(data, pred, threshold=2.0, lm_top=1.0, lm_fp=-0.25):
    good = classify_by_epipolar(data, pred, threshold)
    return lm_top * good + lm_fp * (~good)


def depth_reward(data, pred, threshold=2.0, lm_tp=1.0, lm_fp=-0.25):
    epi_bad = ~classify_by_epipolar(data, pred, threshold)
    good_pairs = classify_by_depth(data, pred, threshold)

    return lm_tp * good_pairs + lm_fp * epi_bad


class DISK(BaseModel):
    default_conf = {
        "detector_type": "rng",  # one of [rng - training, nms - inference]
        "window_size": 8,
        "nms_radius": 2,  # matches with disk nms radius of 5
        "max_num_keypoints": None,
        "force_num_keypoints": False,
        "pad_if_not_divisible": True,
        "detection_threshold": 0.005,
        "desc_dim": 128,
        "arch": {
            "gate": "PReLU",
            "norm": "InstanceNorm2d",
            "upsample": "TrivialUpsample",
            "downsample": "TrivialDownsample",
            "down_block": "ThinDownBlock",  # second option is DownBlock
            "up_block": "ThinUpBlock",  # second option is UpBlock
            # "dropout": False, not used yet
            "bias": True,
            "padding": True,
        },
        "loss": {
            "reward_threshold": 1.5,
            "lambda_tp": 1,
            "lambda_fp": -0.25,
            "lambda_kp": -0.001,
        },
    }

    requred_data_keys = ["image"]

    def _init(self, conf):
        self.set_initialized()

        down_block = get_module(conf.arch.down_block)
        up_block = get_module(conf.arch.up_block)
        kernel_size = 5

        # down dims [3] + [16, 32, 64, 64, 64]
        self.path_down = torch.nn.ModuleList()
        self.path_down.append(
            down_block(3, 16, size=kernel_size, name="down_0", is_first=True, conf=conf)
        )
        self.path_down.append(
            down_block(16, 32, size=kernel_size, name="down_1", conf=conf)
        )
        self.path_down.append(
            down_block(32, 64, size=kernel_size, name="down_2", conf=conf)
        )
        self.path_down.append(
            down_block(64, 64, size=kernel_size, name="down_3", conf=conf)
        )
        self.path_down.append(
            down_block(64, 64, size=kernel_size, name="down_4", conf=conf)
        )

        # up dims = [64, 64, 64, desc_dim+1]
        # bottom dims = [down[-1]] + up       = [64, 64, 64, 64,           desc_dim + 1]
        # horizontal dims = down dims[-2::-1] = [64, 64, 32, 16,           3           ]
        # out dims = up                       = [64, 64, 64, desc_dim + 1              ]
        self.path_up = torch.nn.ModuleList()
        self.path_up.append(
            up_block(64, 64, 64, size=kernel_size, name="up_0", conf=conf)
        )
        self.path_up.append(
            up_block(64, 64, 64, size=kernel_size, name="up_1", conf=conf)
        )
        self.path_up.append(
            up_block(64, 32, 64, size=kernel_size, name="up_2", conf=conf)
        )
        self.path_up.append(
            up_block(
                64, 16, conf.desc_dim + 1, size=kernel_size, name="up_3", conf=conf
            )
        )

        self.train_matcher = ConsistentMatcher(inverse_T=15.0)
        # Turns training off for matcher
        self.train_matcher.requires_grad_(False)
        self.ramp = 0

    def _sample(self, heatmaps):
        v = self.conf.window_size
        device = heatmaps.device
        b, c, h, w = heatmaps.shape

        assert h % v == 0
        assert w % v == 0

        tiled = (
            heatmaps.unfold(2, v, v)
            .unfold(3, v, v)
            .reshape(b, c, h // v, w // v, v * v)
        )

        proposals, accept_mask, logp = point_distribution(tiled.squeeze(1))

        cgrid = torch.stack(
            torch.meshgrid(
                torch.arange(h, device=device),
                torch.arange(w, device=device),
            )[::-1],
            dim=0,
        ).unsqueeze(0)
        cgrid_tiled = (
            cgrid.unfold(2, v, v).unfold(3, v, v).reshape(1, 2, h // v, w // v, v * v)
        )

        xys = (
            torch.gather(
                cgrid_tiled.repeat(b, 1, 1, 1, 1),
                -1,
                proposals.unsqueeze(1).repeat(1, 2, 1, 1).unsqueeze(-1),
            )
            .squeeze(-1)
            .permute(0, 2, 3, 1)
        )

        keypoints = []
        scores = []

        for i in range(b):
            keypoints.append(xys[i][accept_mask[i], :])
            scores.append(logp[i][accept_mask[i]])

        return keypoints, scores

    def _unet(self, images):
        features = [images]
        for block in self.path_down:
            features.append(block(features[-1]))

        f_bot = features[-1]
        features_horizontal = features[-2::-1]
        for i, block in enumerate(self.path_up):
            f_bot = block(f_bot, features_horizontal[i])

        descriptors = f_bot[:, : self.conf.desc_dim]
        heatmaps = f_bot[:, self.conf.desc_dim :]

        return heatmaps, descriptors

    def get_heatmap_and_descriptors(self, data):
        images = data["image"]
        if self.conf.pad_if_not_divisible:
            h, w = images.shape[2:]
            pd_h = 16 - h % 16 if h % 16 > 0 else 0
            pd_w = 16 - w % 16 if w % 16 > 0 else 0
            images = torch.nn.functional.pad(images, (0, pd_w, 0, pd_h), value=0.0)

        heatmaps, descs = self._unet(images)

        return heatmaps, descs

    def _forward(self, data):
        images = data["image"]
        if self.conf.pad_if_not_divisible:
            h, w = images.shape[2:]
            pd_h = 16 - h % 16 if h % 16 > 0 else 0
            pd_w = 16 - w % 16 if w % 16 > 0 else 0
            images = torch.nn.functional.pad(images, (0, pd_w, 0, pd_h), value=0.0)

        heatmaps, descs = self._unet(images)

        if self.conf.detector_type == "rng":
            points, logps = self._sample(heatmaps)
        elif self.conf.detector_type == "nms":
            points, logps = nms(heatmaps, self.conf.nms_radius)
        else:
            raise ValueError(f"Unknown detector type: {self.conf.detector_type}")

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
                mode="random_c",
                bounds=(
                    0,
                    data.get("image_size", torch.tensor(images.shape[-2:]))
                    .min()
                    .item(),
                ),
            )
            scores = pad_and_stack(scores, target_length, -1, mode="zeros")
            descriptors = pad_and_stack(descriptors, target_length, -2, mode="zeros")
        else:
            keypoints = torch.stack(keypoints, 0)
            scores = torch.stack(scores, 0)
            descriptors = torch.stack(descriptors, 0)

        return {
            "keypoints": keypoints,
            "keypoints_score": scores,
            "descriptors": descriptors,
        }

    def _pre_loss_callback(self, seed, epoch):
        if epoch == 0:
            self.ramp = 0
        elif epoch == 1:
            self.ramp = 0.1
        else:
            self.ramp = min(1.0, 0.1 + 0.2 * epoch)

        self.train_matcher.inverse_T = 15.0 + 35.0 * min(1.0, 0.05 * epoch)

    def loss(self, pred, data):
        elementwise_reward = depth_reward(
            data,
            pred,
            threshold=self.conf.loss.reward_threshold,
            lm_tp=self.conf.loss.lambda_tp,
            lm_fp=self.conf.loss.lambda_fp * self.ramp,
        )

        match_dist = self.train_matcher.match_pair(pred)

        with torch.no_grad():
            sample_p = match_dist.dense_p()

        sample_logp = match_dist.dense_logp()

        logp0 = pred["keypoints_score0"]
        logp1 = pred["keypoints_score1"]

        kpts_logp = logp0[:, :, None] + logp1[:, None, :]

        kpts_logp_flat = logp0.sum(dim=1) + logp1.sum(dim=1)

        sample_plogp = sample_p * (sample_logp + kpts_logp)

        reinforce = (elementwise_reward * sample_plogp).sum(dim=(1, 2))
        kp_penalty = self.conf.loss.lambda_kp * kpts_logp_flat

        loss = -reinforce - kp_penalty

        n_keypoints = torch.tensor(
            logp0.shape[-2] + logp1.shape[-2]
        ).repeat(loss.shape[0])
        exp_n_pairs = sample_p.sum(dim=(1, 2))
        exp_reward = (sample_p * elementwise_reward).sum(
            dim=(1, 2)
        ) + self.conf.loss.lambda_kp * n_keypoints

        return {
            "total": loss.sum(),
            "reward": exp_reward,
            "n_keypoints": n_keypoints,
            "n_pairs": exp_n_pairs,
        }, {}
