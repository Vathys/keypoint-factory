import math
from typing import List, Optional, Tuple

import torch

from ...geometry.homography import warp_points_torch
from ...geometry.epipolar import asymm_epipolar_distance_all, T_to_F
from ...geometry.depth import simple_project, unproject


def lscore(dist, thres, type="linear", rescale=True):
    score = None
    if type == "correct":
        score = torch.where(
            torch.isnan(dist), -float("inf"), torch.where(dist < thres, 1, -0.05)
        )
    elif type == "linear":
        score = torch.where(torch.isnan(dist), -float("inf"), 1 - (dist / thres))
    elif type == "fine":
        score = torch.where(
            torch.isnan(dist), -float("inf"), torch.log(thres / (dist + 1e-8)) / thres
        )
    elif type == "coarse":
        score = torch.where(torch.isnan(dist), -float("inf"), 1 - (dist / thres) ** 2)
    else:
        raise RuntimeError(f"Type {type} not found...")

    if rescale:
        score_ = score.clone()
        for b in range(score_.shape[0]):
            if (score_[b] < 0).sum() > 0:
                score_[b][score_[b] < 0] = (
                    score_[b][score_[b] < 0] / score_[b][score_[b] < 0].abs().max()
                )
            if (score_[b] > 0).sum() > 0:
                score_[b][score_[b] > 0] = (
                    score_[b][score_[b] > 0] / score_[b][score_[b] > 0].abs().max()
                )
        score_ = torch.where(torch.isnan(score_), -1, score_)
        return score_
    else:
        return score


def and_mult(a, b, rescale=True):
    score = torch.mul(a.abs(), b.abs()) * torch.where(
        torch.logical_and(a > 0, b > 0), 1, -1
    )

    if rescale:
        score_ = score.clone()
        for b in range(score_.shape[0]):
            if (score_[b] < 0).sum() > 0:
                score_[b][score_[b] < 0] = (
                    score_[b][score_[b] < 0] / score_[b][score_[b] < 0].abs().max()
                )
            if (score_[b] > 0).sum() > 0:
                score_[b][score_[b] > 0] = (
                    score_[b][score_[b] > 0] / score_[b][score_[b] > 0].abs().max()
                )
        return score_
    else:
        return score


def to_sequence(map):
    return map.flatten(-2).transpose(-1, -2)


def to_map(sequence):
    n = sequence.shape[-2]
    e = math.isqrt(n)
    assert e * e == n
    assert e * e == n
    sequence.transpose(-1, -2).unflatten(-1, [e, e])


def pad_to_length(
    x,
    length: int,
    pad_dim: int = -2,
    mode: str = "zeros",  # zeros, ones, random, random_c
    bounds: Tuple[int] = (None, None),
    constant: float = 0.0,
):
    shape = list(x.shape)
    d = x.shape[pad_dim]
    assert d <= length
    if d == length:
        return x
    shape[pad_dim] = length - d

    low, high = bounds

    if mode == "zeros":
        xn = torch.zeros(*shape, device=x.device, dtype=x.dtype)
    elif mode == "ones":
        xn = torch.ones(*shape, device=x.device, dtype=x.dtype)
    elif mode == "constant":
        xn = torch.full(shape, constant, device=x.device, dtype=x.dtype)
    elif mode == "random":
        low = low if low is not None else x.min()
        high = high if high is not None else x.max()
        xn = torch.empty(*shape, device=x.device).uniform_(low, high)
    elif mode == "random_c":
        low, high = bounds  # we use the bounds as fallback for empty seq.
        xn = torch.cat(
            [
                torch.empty(*shape[:-1], 1, device=x.device).uniform_(
                    x[..., i].min() if d > 0 else low,
                    x[..., i].max() if d > 0 else high,
                )
                for i in range(shape[-1])
            ],
            dim=-1,
        )
    else:
        raise ValueError(mode)
    return torch.cat([x, xn], dim=pad_dim)


def pad_and_stack(
    sequences: List[torch.Tensor],
    length: Optional[int] = None,
    pad_dim: int = -2,
    **kwargs,
):
    if length is None:
        length = max([x.shape[pad_dim] for x in sequences])

    y = torch.stack([pad_to_length(x, length, pad_dim, **kwargs) for x in sequences], 0)
    return y


def cut_to_match(reference, t, n_pref=2):
    """
    Slice tensor `t` along spatial dimensions to match `reference`, by
    picking the central region. Ignores first `n_pref` axes
    """

    if reference.shape[n_pref:] == t.shape[n_pref:]:
        # sizes match, no slicing necessary
        return t

    # compute the difference along all spatial axes
    diffs = [s - r for s, r in zip(t.shape[n_pref:], reference.shape[n_pref:])]

    # check if diffs are even, which is necessary if we want a truly centered crop
    if not all(d % 2 == 0 for d in diffs) and all(d >= 0 for d in diffs):
        fmt = "Tried to slice `t` of size {} to match `reference` of size {}"
        msg = fmt.format(t.shape, reference.shape)
        raise RuntimeError(msg)

    # pick the full extent of `batch` and `feature` axes
    slices = [slice(None, None)] * n_pref

    # for all remaining pick between diff//2 and size-diff//2
    for d in diffs:
        if d > 0:
            slices.append(slice(d // 2, -(d // 2)))
        elif d == 0:
            slices.append(slice(None, None))

    if slices == []:
        return t
    else:
        return t[slices]


def size_is_pow2(t):
    """Check if the trailing spatial dimensions are powers of 2"""
    return all(s % 2 == 0 for s in t.size()[-2:])


def tile(hm, window):
    b, c, h, w = hm.shape

    assert hm.shape[2] % window == 0
    assert hm.shape[3] % window == 0

    return (
        hm.unfold(2, window, window)
        .unfold(3, window, window)
        .reshape(b, c, h // window, w // window, window * window)
    )


def select_on_last(values, indices):
    return torch.gather(values, -1, indices[..., None]).squeeze(-1)


def distance_matrix(fs1, fs2):
    """
    fs1: B x N x F
    fs2: B x M x F
    Assumes fs1 and fs2 are normalized!
    returns distance matrix of size B x N x M
    """
    if fs1.shape[1] == 0:
        fs1 = torch.zeros((fs1.shape[0], 1, fs1.shape[2]), device=fs1.device)
    if fs2.shape[1] == 0:
        fs2 = torch.zeros((fs2.shape[0], 1, fs2.shape[2]), device=fs2.device)
    dist = torch.einsum("...if,...jf->...ij", fs1, fs2)
    return 1.414213 * (1.0 - dist).clamp(min=1e-6).sqrt()


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


class CycleMatcher:
    def __init__(self, threshold):
        self.threshold = threshold

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

        dist0 = torch.norm(diff0, p=2, dim=-1)
        dist1 = torch.norm(diff1, p=2, dim=-1)

        dist = torch.min(dist0, dist1)
        mask_visible = ~torch.isnan(dist)
        inf = dist.new_tensor(float("inf"))
        dist = torch.where(mask_visible, dist, inf)

        min0 = dist.min(-1).indices
        min1 = dist.min(-2).indices
        ismin0 = torch.zeros(dist.shape, dtype=torch.bool, device=dist.device)
        ismin1 = ismin0.clone()

        ismin0.scatter_(-1, min0.unsqueeze(-1), value=1)
        ismin1.scatter_(-2, min1.unsqueeze(-2), value=1)
        positive = ismin0 & ismin1 & (dist < self.threshold**2)
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
        positive = ismin0 & ismin1 & (dist < self.threshold**2)
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

    def match_by_epipolar(self, data, pred):
        F_0_1 = T_to_F(data["view0"]["camera"], data["view1"]["camera"], data["T_0to1"])
        F_1_0 = T_to_F(data["view1"]["camera"], data["view0"]["camera"], data["T_1to0"])

        kpts0 = pred["keypoints0"]
        kpts1 = pred["keypoints1"]

        epi_0_1 = asymm_epipolar_distance_all(kpts0, kpts1, F_0_1).abs()
        epi_1_0 = asymm_epipolar_distance_all(kpts1, kpts0, F_1_0).abs()

        dist = torch.max(epi_1_0, epi_0_1.transpose(-1, -2))
        mask_visible = ~torch.isnan(dist)
        inf = dist.new_tensor(float("inf"))
        dist = torch.where(mask_visible, dist, inf)

        min0 = dist.min(-1).indices
        min1 = dist.min(-2).indices
        ismin0 = torch.zeros(dist.shape, dtype=torch.bool, device=epi_0_1.device)
        ismin1 = ismin0.clone()

        ismin0.scatter_(-1, min0.unsqueeze(-1), value=1)
        ismin1.scatter_(-2, min1.unsqueeze(-2), value=1)
        positive = ismin0 & ismin1 & (dist < self.threshold**2)
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
