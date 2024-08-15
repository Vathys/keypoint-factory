import matplotlib as mpl
import torch

from ..models.extractors.diskv2 import reproject_homography
from ..models.utils.metrics import compute_correctness
from ..models.utils.misc import lscore
from ..utils.tensor import batch_to_device
from .viz2d import (
    cm_ranking,
    cm_RdGn,
    plot_heatmaps,
    plot_image_grid,
    plot_keypoints,
    plot_matches,
)

plasma = mpl.colormaps["plasma"]


def make_reward_figures(pred_, data_, n_pairs=2):
    if "0to1" in pred_.keys():
        pred_ = pred_["0to1"]

    images, kpts, colors = [], [], []

    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)

    view0, view1 = data["view0"], data["view1"]

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    kpscore0, kpscore1 = pred["keypoint_scores0"], pred["keypoint_scores1"]

    for i in range(n_pairs):
        images.append(
            [view0["image"][i].permute(1, 2, 0), view1["image"][i].permute(1, 2, 0)]
        )
        kpts.append([kp0[i], kp1[i]])

        H_0to1 = data["H_0to1"]

        kpts0_r = reproject_homography(
            kp0, H_0to1, *data["view1"]["image"].shape[2:], False
        )
        kpts1_r = reproject_homography(
            kp1, H_0to1, *data["view0"]["image"].shape[2:], True
        )

        diff0 = kp0[:, :, None, :] - kpts1_r[:, None, :, :]
        diff1 = kp1[:, None, :, :] - kpts0_r[:, :, None, :]

        dist0 = torch.norm(diff0, p=2, dim=-1)
        dist1 = torch.norm(diff1, p=2, dim=-1)

        reproj_error0 = torch.min(dist0.nan_to_num(nan=float("inf")), dim=-1).values
        reproj_error1 = torch.min(dist1.nan_to_num(nan=float("inf")), dim=-2).values

        threshold = 3.0
        type = "linear"

        score0 = lscore(reproj_error0, threshold, type=type)
        score1 = lscore(reproj_error1, threshold, type=type)

        score0[torch.all(kpts0_r.isnan(), dim=-1)] = 0
        score1[torch.all(kpts1_r.isnan(), dim=-1)] = 0

        reward0 = score0 * kpscore0
        reward1 = score1 * kpscore1

        colors.append([plasma(reward0[i]), plasma(reward1[i])])

    fig, axes = plot_image_grid(images, return_fig=True, set_lim=True)
    [
        plot_keypoints(kpts[i], axes=axes[i], colors=colors[i], ps=1)
        for i in range(n_pairs)
    ]

    return {"reward": fig}


def make_score_figures(pred_, data_, n_pairs=2):
    if "0to1" in pred_.keys():
        pred_ = pred_["0to1"]

    images, kpts, colors = [], [], []

    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)

    view0, view1 = data["view0"], data["view1"]

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    kpscore0, kpscore1 = pred["keypoint_scores0"], pred["keypoint_scores1"]

    for i in range(n_pairs):
        images.append(
            [view0["image"][i].permute(1, 2, 0), view1["image"][i].permute(1, 2, 0)]
        )
        kpts.append([kp0[i], kp1[i]])

        colors.append([cm_ranking(kpscore0[i]), cm_ranking(kpscore1[i])])

    fig, axes = plot_image_grid(images, return_fig=True, set_lim=True)
    [
        plot_keypoints(kpts[i], axes=axes[i], colors=colors[i], ps=1)
        for i in range(n_pairs)
    ]

    return {"score": fig}


def make_correct_figures(pred_, data_, n_pairs=2):
    if "0to1" in pred_.keys():
        pred_ = pred_["0to1"]

    images, kpts, colors = [], [], []

    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)

    view0, view1 = data["view0"], data["view1"]

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]

    for i in range(n_pairs):
        images.append(
            [view0["image"][i].permute(1, 2, 0), view1["image"][i].permute(1, 2, 0)]
        )
        kpts.append([kp0[i], kp1[i]])
        metrics = compute_correctness(data, pred)
        correct0 = metrics["correct0"]
        correct1 = metrics["correct1"]

        colors.append([cm_RdGn(correct0[i]), cm_RdGn(correct1[i])])

    fig, axes = plot_image_grid(images, return_fig=True, set_lim=True)
    [
        plot_keypoints(kpts[i], axes=axes[i], colors=colors[i], ps=1)
        for i in range(n_pairs)
    ]

    return {"correct": fig}


def make_hm_plot(pred_, data_, n_pairs=2):
    # print first n pairs in batch
    if "0to1" in pred_.keys():
        pred_ = pred_["0to1"]
    images, kpts = [], []
    heatmaps = []
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)

    view0, view1 = data["view0"], data["view1"]

    n_pairs = min(n_pairs, view0["image"].shape[0])
    assert view0["image"].shape[0] >= n_pairs

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]

    for i in range(n_pairs):
        images.append(
            [view0["image"][i].permute(1, 2, 0), view1["image"][i].permute(1, 2, 0)]
        )
        kpts.append([kp0[i], kp1[i]])

        if "heatmap0" in pred.keys():
            heatmaps.append(
                [
                    torch.sigmoid(pred["heatmap0"][i, 0]),
                    torch.sigmoid(pred["heatmap1"][i, 0]),
                ]
            )

    fig, axes = plot_image_grid(images, return_fig=True, set_lim=True)
    if len(heatmaps) > 0:
        [plot_heatmaps(heatmaps[i], axes=axes[i], a=0.7) for i in range(n_pairs)]
    [
        plot_keypoints(kpts[i], axes=axes[i], colors="royalblue", ps=1)
        for i in range(n_pairs)
    ]

    return {"keypoints": fig}


def make_keypoint_figures(pred_, data_, n_pairs=2):
    res = {
        **make_hm_plot(pred_, data_, n_pairs),
        **make_correct_figures(pred_, data_, n_pairs),
        **make_reward_figures(pred_, data_, n_pairs),
        **make_score_figures(pred_, data_, n_pairs),
    }
    return res


def make_match_figures(pred_, data_, n_pairs=2):
    # print first n pairs in batch
    if "0to1" in pred_.keys():
        pred_ = pred_["0to1"]
    images, kpts, matches, mcolors = [], [], [], []
    heatmaps = []
    pred = batch_to_device(pred_, "cpu", non_blocking=False)
    data = batch_to_device(data_, "cpu", non_blocking=False)

    view0, view1 = data["view0"], data["view1"]

    n_pairs = min(n_pairs, view0["image"].shape[0])
    assert view0["image"].shape[0] >= n_pairs

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0 = pred["matches0"]
    gtm0 = pred["gt_matches0"]

    for i in range(n_pairs):
        valid = (m0[i] > -1) & (gtm0[i] >= -1)
        kpm0, kpm1 = kp0[i][valid].numpy(), kp1[i][m0[i][valid]].numpy()
        images.append(
            [view0["image"][i].permute(1, 2, 0), view1["image"][i].permute(1, 2, 0)]
        )
        kpts.append([kp0[i], kp1[i]])
        matches.append((kpm0, kpm1))

        correct = gtm0[i][valid] == m0[i][valid]

        if "heatmap0" in pred.keys():
            heatmaps.append(
                [
                    torch.sigmoid(pred["heatmap0"][i, 0]),
                    torch.sigmoid(pred["heatmap1"][i, 0]),
                ]
            )
        elif "depth" in view0.keys() and view0["depth"] is not None:
            heatmaps.append([view0["depth"][i], view1["depth"][i]])

        mcolors.append(cm_RdGn(correct).tolist())

    fig, axes = plot_image_grid(images, return_fig=True, set_lim=True)
    if len(heatmaps) > 0:
        [plot_heatmaps(heatmaps[i], axes=axes[i], a=1.0) for i in range(n_pairs)]
    [plot_keypoints(kpts[i], axes=axes[i], colors="royalblue") for i in range(n_pairs)]
    [
        plot_matches(*matches[i], color=mcolors[i], axes=axes[i], a=0.5, lw=1.0, ps=0.0)
        for i in range(n_pairs)
    ]

    return {"matching": fig}
