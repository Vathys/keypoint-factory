import torch

from ...geometry.depth import project, sample_depth
from ...geometry.homography import warp_points_torch


@torch.no_grad()
def matcher_metrics(pred, data, prefix="", prefix_gt=None):
    def recall(m, gt_m):
        mask = (gt_m > -1).float()
        return ((m == gt_m) * mask).sum(1) / (1e-8 + mask.sum(1))

    def accuracy(m, gt_m):
        mask = (gt_m >= -1).float()
        return ((m == gt_m) * mask).sum(1) / (1e-8 + mask.sum(1))

    def precision(m, gt_m):
        mask = ((m > -1) & (gt_m >= -1)).float()
        return ((m == gt_m) * mask).sum(1) / (1e-8 + mask.sum(1))

    def ranking_ap(m, gt_m, scores):
        p_mask = ((m > -1) & (gt_m >= -1)).float()
        r_mask = (gt_m > -1).float()
        sort_ind = torch.argsort(-scores)
        sorted_p_mask = torch.gather(p_mask, -1, sort_ind)
        sorted_r_mask = torch.gather(r_mask, -1, sort_ind)
        sorted_tp = torch.gather(m == gt_m, -1, sort_ind)
        p_pts = torch.cumsum(sorted_tp * sorted_p_mask, -1) / (
            1e-8 + torch.cumsum(sorted_p_mask, -1)
        )
        r_pts = torch.cumsum(sorted_tp * sorted_r_mask, -1) / (
            1e-8 + sorted_r_mask.sum(-1)[:, None]
        )
        r_pts_diff = r_pts[..., 1:] - r_pts[..., :-1]
        return torch.sum(r_pts_diff * p_pts[:, None, -1], dim=-1)

    if prefix_gt is None:
        prefix_gt = prefix
    rec = recall(pred[f"{prefix}matches0"], data[f"gt_{prefix_gt}matches0"])
    prec = precision(pred[f"{prefix}matches0"], data[f"gt_{prefix_gt}matches0"])
    acc = accuracy(pred[f"{prefix}matches0"], data[f"gt_{prefix_gt}matches0"])
    ap = ranking_ap(
        pred[f"{prefix}matches0"],
        data[f"gt_{prefix_gt}matches0"],
        pred[f"{prefix}matching_scores0"],
    )
    metrics = {
        f"{prefix}match_recall": rec,
        f"{prefix}match_precision": prec,
        f"{prefix}accuracy": acc,
        f"{prefix}average_precision": ap,
    }
    return metrics


def compute_correctness(data, pred, thresh=3.0):
    kpts0 = pred["keypoints0"]  # [B, N, 2]
    kpts1 = pred["keypoints1"]  # [B, M, 2]

    if "depth" in data["view0"]:
        depth0 = data["view0"]["depth"]
        depth1 = data["view1"]["depth"]
        cam0 = data["view0"]["camera"]
        cam1 = data["view1"]["camera"]
        T0_1 = data["T_0to1"]
        T1_0 = data["T_1to0"]

        d0, valid0 = sample_depth(kpts0, depth0)
        d1, valid1 = sample_depth(kpts1, depth1)

        kpts0_1, visible0 = project(
            kpts0, d0, depth1, cam0, cam1, T0_1, valid0, 3.0
        )  # [B, M, 2]
        kpts1_0, visible1 = project(
            kpts1, d1, depth0, cam1, cam0, T1_0, valid1, 3.0
        )  # [B, N, 2]
        mask_visible = visible0.unsqueeze(-1) & visible1.unsqueeze(-2)

        dist0 = torch.norm(kpts0_1[:, :, None, :] - kpts1[:, None, :, :], dim=-1)
        dist1 = torch.norm(kpts0[:, :, None, :] - kpts1_0[:, None, :, :], dim=-1)
        dist = torch.max(dist0, dist1)
        inf = dist.new_tensor(float("inf"))
        dist = torch.where(mask_visible, dist, inf)

        correct0 = dist.min(-1).values < thresh
        correct1 = dist.min(-2).values < thresh
    else:
        H = data["H_0to1"]

        kpts0_1 = warp_points_torch(kpts0.double(), H.double(), inverse=False)
        kpts1_0 = warp_points_torch(kpts1.double(), H.double(), inverse=True)

        dist0 = torch.norm(kpts0_1[:, :, None, :] - kpts1[:, None, :, :], dim=-1)
        dist1 = torch.norm(kpts0[:, :, None, :] - kpts1_0[:, None, :, :], dim=-1)
        dist = torch.max(dist0, dist1)

        correct0 = dist.min(-1).values < thresh
        correct1 = dist.min(-2).values < thresh

    return {
        "correct0": correct0,
        "correct1": correct1,
        "dist0": dist0,
        "dist1": dist1,
    }
