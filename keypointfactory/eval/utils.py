import numpy as np
import torch
from kornia.geometry.homography import find_homography_dlt
from scipy.stats import pearsonr
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)  # implement roc_auc_score and average_precision on your own?

# ask about AUCMetrics class

from ..geometry.epipolar import generalized_epi_dist, relative_pose_error
from ..geometry.gt_generation import IGNORE_FEATURE
from ..geometry.homography import (
    homography_corner_error,
    sym_homography_error,
    warp_points_torch,
)
from ..geometry.utils import div0
from ..robust_estimators import load_estimator
from ..utils.tensor import index_batch
from ..utils.tools import AUCMetric


def check_keys_recursive(d, pattern):
    if isinstance(pattern, dict):
        {check_keys_recursive(d[k], v) for k, v in pattern.items()}
    else:
        for k in pattern:
            assert k in d.keys()


def compute_correctness(kpts1, kpts2, kpts1_w, kpts2_w, thresh, mutual=True):

    def compute_correctness_single(kpts, kpts_w):
        dist = torch.norm(kpts_w[:, None] - kpts[None], dim=-1)
        min_dist, matches = dist.min(dim=1)
        correct = min_dist <= thresh
        if mutual:
            idx = dist.argmin(dim=1)
            idx_w = dist.argmin(dim=0)
            correct &= torch.eq(torch.arange(len(kpts_w)), idx_w[idx])
        return min_dist, matches, correct

    dist1, matches1, correct1 = compute_correctness_single(kpts2, kpts1_w)
    dist2, matches2, correct2 = compute_correctness_single(kpts1, kpts2_w)
    return {
        "correct0": correct1,
        "correct1": correct2,
        "dist0": dist1,
        "dist1": dist2,
        "matches0": matches1,
        "matches1": matches2,
    }


def get_matches_scores(kpts0, kpts1, matches0, mscores0):
    m0 = matches0 > -1
    m1 = matches0[m0]
    pts0 = kpts0[m0]
    pts1 = kpts1[m1]
    scores = mscores0[m0]
    return pts0, pts1, scores


def eval_per_batch_item(data: dict, pred: dict, eval_f, *args, **kwargs):
    # Batched data
    results = [
        eval_f(data_i, pred_i, *args, **kwargs)
        for data_i, pred_i in zip(index_batch(data), index_batch(pred))
    ]
    # Return a dictionary of lists with the evaluation of each item
    return {k: [r[k] for r in results] for k in results[0].keys()}


def eval_keypoints_homography(
    data, pred, correctness_thresh=3.0, top_kpts=200, mutual=True
):
    check_keys_recursive(data, ["H_0to1"])
    check_keys_recursive(pred, ["keypoints0", "keypoints1", "matches0", "matches1"])

    H_gt = data["H_0to1"]
    if H_gt.ndim > 2:
        return eval_per_batch_item(data, pred, eval_keypoints_homography)

    image0 = data["view0"]["image"].permute(1, 2, 0)
    image1 = data["view1"]["image"].permute(1, 2, 0)

    kpts0 = pred["keypoints0"]
    kpts1 = pred["keypoints1"]
    kpts_scores0 = pred["keypoint_scores0"]
    kpts_scores1 = pred["keypoint_scores1"]

    valid_matches0 = pred["matches0"] > 0
    valid_matches1 = pred["matches1"] > 0

    kpts0_1 = warp_points_torch(kpts0, H_gt, inverse=False)
    kpts1_0 = warp_points_torch(kpts1, H_gt, inverse=True)

    vis0 = torch.all(
        (kpts0_1 >= 0) & (kpts0_1 <= torch.tensor(image1.shape[:2][::-1])), dim=-1
    )
    vis1 = torch.all(
        (kpts1_0 >= 0) & (kpts1_0 <= torch.tensor(image0.shape[:2][::-1])), dim=-1
    )

    cor_dict = compute_correctness(
        kpts0, kpts1, kpts0_1, kpts1_0, thresh=correctness_thresh, mutual=mutual
    )
    dist0 = cor_dict["dist0"]
    dist1 = cor_dict["dist1"]
    correct0 = cor_dict["correct0"]
    correct1 = cor_dict["correct1"]
    ddescriptors0 = torch.norm(
        pred["descriptors0"] - pred["descriptors1"][cor_dict["matches0"]], p=2, dim=1
    )
    ddescriptors1 = torch.norm(
        pred["descriptors1"] - pred["descriptors0"][cor_dict["matches1"]], p=2, dim=1
    )
    indices0 = kpts_scores0[vis0].sort(descending=True)[1][:top_kpts]
    indices1 = kpts_scores1[vis1].sort(descending=True)[1][:top_kpts]
    pred0, pred1 = torch.zeros_like(
        kpts_scores0[vis0], dtype=torch.bool
    ), torch.zeros_like(kpts_scores1[vis1], dtype=torch.bool)
    pred0[indices0] = True
    pred1[indices1] = True

    results = {}

    roc_auc0 = roc_auc_score(correct0[vis0], kpts_scores0[vis0])
    roc_auc1 = roc_auc_score(correct1[vis1], kpts_scores1[vis1])

    results["roc_auc"] = (roc_auc0 + roc_auc1) / 2

    avg_prec0 = average_precision_score(correct0[vis0], kpts_scores0[vis0])
    avg_prec1 = average_precision_score(correct1[vis1], kpts_scores1[vis1])

    results["avg_prec"] = (avg_prec0 + avg_prec1) / 2

    prec0 = ((correct0[vis0] & pred0).sum().float() / pred0.sum()).float().item()
    prec1 = ((correct1[vis1] & pred1).sum().float() / pred1.sum()).float().item()

    results[f"prec0@{top_kpts}pts"] = (prec0 + prec1) / 2

    rec0 = (
        ((correct0[vis0] & pred0).sum().float() / correct0[vis0].sum()).float().item()
    )
    rec1 = (
        ((correct1[vis1] & pred1).sum().float() / correct1[vis1].sum()).float().item()
    )

    results[f"recall0@{top_kpts}pts"] = (rec0 + rec1) / 2

    results["loc_err"] = (
        dist0[vis0 & correct0].mean() + dist1[vis1 & correct1].mean()
    ) / 2

    # Is there a way to combine the p-values?
    # Look at Fisher's Score.
    dd_corr0, dd_corr_p0 = pearsonr(torch.log(dist0[vis0]), ddescriptors0[vis0])

    dd_corr1, dd_corr_p1 = pearsonr(torch.log(dist1[vis1]), ddescriptors1[vis1])

    results["ddescriptor_corr"] = (dd_corr0 + dd_corr1) / 2

    results["repeatability"] = (
        div0(correct0[vis0].sum() + correct1[vis1].sum(), vis0.sum() + vis1.sum())
        .float()
        .item()
    )
    results["num_correct"] = (
        correct0[vis0].sum().item() + correct1[vis1].sum().item()
    ) / 2
    return results


def eval_matches_epipolar(data: dict, pred: dict) -> dict:
    check_keys_recursive(data, ["view0", "view1", "T_0to1"])
    check_keys_recursive(
        pred, ["keypoints0", "keypoints1", "matches0", "matching_scores0"]
    )

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0, scores0 = pred["matches0"], pred["matching_scores0"]
    pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)

    results = {}

    # match metrics
    n_epi_err = generalized_epi_dist(
        pts0[None],
        pts1[None],
        data["view0"]["camera"],
        data["view1"]["camera"],
        data["T_0to1"],
        False,
        essential=True,
    )[0]
    results["epi_prec@1e-4"] = (n_epi_err < 1e-4).float().mean()
    results["epi_prec@5e-4"] = (n_epi_err < 5e-4).float().mean()
    results["epi_prec@1e-3"] = (n_epi_err < 1e-3).float().mean()

    results["num_matches"] = pts0.shape[0]
    results["num_keypoints"] = (kp0.shape[0] + kp1.shape[0]) / 2.0

    return results


def eval_matches_homography(data: dict, pred: dict) -> dict:
    check_keys_recursive(data, ["H_0to1"])
    check_keys_recursive(
        pred, ["keypoints0", "keypoints1", "matches0", "matching_scores0"]
    )

    H_gt = data["H_0to1"]
    if H_gt.ndim > 2:
        return eval_per_batch_item(data, pred, eval_matches_homography)

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0, scores0 = pred["matches0"], pred["matching_scores0"]
    pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)
    err = sym_homography_error(pts0, pts1, H_gt)
    results = {}
    results["prec@1px"] = (err < 1).float().mean().nan_to_num().item()
    results["prec@3px"] = (err < 3).float().mean().nan_to_num().item()
    results["num_matches"] = pts0.shape[0]
    results["num_keypoints"] = (kp0.shape[0] + kp1.shape[0]) / 2.0
    return results


def eval_relative_pose_robust(data, pred, conf):
    check_keys_recursive(data, ["view0", "view1", "T_0to1"])
    check_keys_recursive(
        pred, ["keypoints0", "keypoints1", "matches0", "matching_scores0"]
    )

    T_gt = data["T_0to1"]
    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0, scores0 = pred["matches0"], pred["matching_scores0"]
    pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)

    results = {}

    estimator = load_estimator("relative_pose", conf["estimator"])(conf)
    data_ = {
        "m_kpts0": pts0,
        "m_kpts1": pts1,
        "camera0": data["view0"]["camera"][0],
        "camera1": data["view1"]["camera"][0],
    }
    est = estimator(data_)

    if not est["success"]:
        results["rel_pose_error"] = float("inf")
        results["ransac_inl"] = 0
        results["ransac_inl%"] = 0
    else:
        # R, t, inl = ret
        M = est["M_0to1"]
        inl = est["inliers"].numpy()
        t_error, r_error = relative_pose_error(T_gt, M.R, M.t)
        results["rel_pose_error"] = max(r_error, t_error)
        results["ransac_inl"] = np.sum(inl)
        results["ransac_inl%"] = np.mean(inl)

    return results


def eval_homography_robust(data, pred, conf):
    H_gt = data["H_0to1"]
    if H_gt.ndim > 2:
        return eval_per_batch_item(data, pred, eval_relative_pose_robust, conf)

    estimator = load_estimator("homography", conf["estimator"])(conf)

    data_ = {}
    if "keypoints0" in pred:
        kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
        m0, scores0 = pred["matches0"], pred["matching_scores0"]
        pts0, pts1, _ = get_matches_scores(kp0, kp1, m0, scores0)
        data_["m_kpts0"] = pts0
        data_["m_kpts1"] = pts1
    if "lines0" in pred:
        if "orig_lines0" in pred:
            lines0 = pred["orig_lines0"]
            lines1 = pred["orig_lines1"]
        else:
            lines0 = pred["lines0"]
            lines1 = pred["lines1"]
        m_lines0, m_lines1, _ = get_matches_scores(
            lines0, lines1, pred["line_matches0"], pred["line_matching_scores0"]
        )
        data_["m_lines0"] = m_lines0
        data_["m_lines1"] = m_lines1

    est = estimator(data_)
    if est["success"]:
        M = est["M_0to1"]
        error_r = homography_corner_error(M, H_gt, data["view0"]["image_size"]).item()
    else:
        error_r = float("inf")

    results = {}
    results["H_error_ransac"] = error_r
    if "inliers" in est:
        inl = est["inliers"]
        results["ransac_inl"] = inl.float().sum().item()
        results["ransac_inl%"] = inl.float().sum().item() / max(len(inl), 1)

    return results


def eval_homography_dlt(data, pred):
    H_gt = data["H_0to1"]
    H_inf = torch.ones_like(H_gt) * float("inf")

    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
    m0, scores0 = pred["matches0"], pred["matching_scores0"]
    pts0, pts1, scores = get_matches_scores(kp0, kp1, m0, scores0)
    scores = scores.to(pts0)
    results = {}
    try:
        if H_gt.ndim == 2:
            pts0, pts1, scores = pts0[None], pts1[None], scores[None]
        h_dlt = find_homography_dlt(pts0, pts1, scores)
        if H_gt.ndim == 2:
            h_dlt = h_dlt[0]
    except AssertionError:
        h_dlt = H_inf

    error_dlt = homography_corner_error(h_dlt, H_gt, data["view0"]["image_size"])
    results["H_error_dlt"] = error_dlt.item()
    return results


def eval_poses(pose_results, auc_ths, key, unit="°"):
    pose_aucs = {}
    best_th = -1
    for th, results_i in pose_results.items():
        pose_aucs[th] = AUCMetric(auc_ths, results_i[key]).compute()
    mAAs = {k: np.mean(v) for k, v in pose_aucs.items()}
    best_th = max(mAAs, key=mAAs.get)

    if len(pose_aucs) > -1:
        print("Tested ransac setup with following results:")
        print("AUC", pose_aucs)
        print("mAA", mAAs)
        print("best threshold =", best_th)

    summaries = {}

    for i, ath in enumerate(auc_ths):
        summaries[f"{key}@{ath}{unit}"] = pose_aucs[best_th][i]
    summaries[f"{key}_mAA"] = mAAs[best_th]

    for k, v in pose_results[best_th].items():
        arr = np.array(v)
        if not np.issubdtype(np.array(v).dtype, np.number):
            continue
        summaries[f"m{k}"] = round(np.median(arr), 3)
    return summaries, best_th


def get_tp_fp_pts(pred_matches, gt_matches, pred_scores):
    """
    Computes the True Positives (TP), False positives (FP), the score associated
    to each match and the number of positives for a set of matches.
    """
    assert pred_matches.shape == pred_scores.shape
    ignore_mask = gt_matches != IGNORE_FEATURE
    pred_matches, gt_matches, pred_scores = (
        pred_matches[ignore_mask],
        gt_matches[ignore_mask],
        pred_scores[ignore_mask],
    )
    num_pos = np.sum(gt_matches != -1)
    pred_positives = pred_matches != -1
    tp = pred_matches[pred_positives] == gt_matches[pred_positives]
    fp = pred_matches[pred_positives] != gt_matches[pred_positives]
    scores = pred_scores[pred_positives]
    return tp, fp, scores, num_pos


def AP(tp, fp):
    recall = tp
    precision = tp / np.maximum(tp + fp, 1e-9)
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for i in range(precision.size - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])
    i = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])
    return ap


def aggregate_pr_results(results, suffix=""):
    tp_list = np.concatenate(results["tp" + suffix], axis=0)
    fp_list = np.concatenate(results["fp" + suffix], axis=0)
    scores_list = np.concatenate(results["scores" + suffix], axis=0)
    n_gt = max(results["num_pos" + suffix], 1)

    out = {}
    idx = np.argsort(scores_list)[::-1]
    tp_vals = np.cumsum(tp_list[idx]) / n_gt
    fp_vals = np.cumsum(fp_list[idx]) / n_gt
    out["curve_recall" + suffix] = tp_vals
    out["curve_precision" + suffix] = tp_vals / np.maximum(tp_vals + fp_vals, 1e-9)
    out["AP" + suffix] = AP(tp_vals, fp_vals) * 100
    return out
