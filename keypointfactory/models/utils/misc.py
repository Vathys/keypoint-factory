import math
from typing import List, Optional, Tuple

import torch

from ...geometry.homography import warp_points_torch


def lscore(dist, thres, type="linear"):
    score = None
    if type == "correct":
        score = torch.where(
            torch.isnan(dist), -float("inf"), torch.where(dist < thres, 1, -0.05)
        )
    elif type == "linear":
        score = torch.where(
            torch.isnan(dist),
            -float("inf"),
            1 - (dist / thres)
        )
    elif type == "fine":
        score = torch.where(
            torch.isnan(dist),
            -float("inf"),
            torch.log(thres / (dist + 1e-8)) / thres
        )
    elif type == "coarse":
        score = torch.where(
            torch.isnan(dist),
            -float("inf"),
            1 - (dist / thres) ** 2
        )
    else:
        raise RuntimeError(f"Type {type} not found...")

    score_ = score.clone()
    for b in range(score_.shape[0]):
        if (score_[b] < 0).sum() > 0:
            score_[b][score_[b] < 0] = score_[b][score_[b] < 0] / score_[b][score_[b] < 0].abs().max()
        if (score_[b] > 0).sum() > 0:
            score_[b][score_[b] > 0] = score_[b][score_[b] > 0] / score_[b][score_[b] > 0].abs().max()
    
    return score_

def and_mult(a, b):
    return torch.mul(a.abs(), b.abs()) * torch.where(
        torch.logical_and(a > 0, b > 0), 1, -1
    )


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


def unproject(kpts, depth, cam, pose):
    batch, h, w = depth.shape
    in_range = (
        (0 <= kpts[:, :, 0])
        & (kpts[:, :, 0] < w)
        & (0 <= kpts[:, :, 1])
        & (kpts[:, :, 1] < h)
    )
    finite = torch.isfinite(kpts).all(dim=2)
    valid_depth = in_range & finite

    valid_kpts = []
    for b in range(batch):
        valid_kpts.append(kpts[b, valid_depth[b, :], :].to(torch.int64))
    fdepth = torch.full(
        kpts.shape[:2],
        fill_value=float("NaN"),
        device=kpts.device,
        dtype=kpts.dtype,
    )

    for b in range(batch):
        fdepth[b, valid_depth[b]] = depth[b, valid_kpts[b][:, 1], valid_kpts[b][:, 0]]

    kptsc = cam.image2cam(kpts) * fdepth.unsqueeze(-1).repeat(1, 1, 3)
    kptsw = pose.inv().transform(kptsc)

    return kptsw


def project(kptsw, cam, pose):
    ext = pose.transform(kptsw)
    kpts, _ = cam.cam2image(ext)
    return kpts


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
