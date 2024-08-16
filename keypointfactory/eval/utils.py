import numpy as np
import pandas as pd
import torch
from kornia.geometry.homography import find_homography_dlt

from ..geometry.depth import project, sample_depth
from ..geometry.epipolar import generalized_epi_dist, relative_pose_error
from ..geometry.gt_generation import IGNORE_FEATURE
from ..geometry.homography import (
    homography_corner_error,
    sym_homography_error,
    warp_points_torch,
)
from ..geometry.utils import div0
from ..models.utils.misc import distance_matrix
from ..robust_estimators import load_estimator
from ..utils.tensor import index_batch, map_tensor
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
        if dist.shape[0] == 0 or dist.shape[1] == 0:
            return (
                torch.Tensor(dist.shape),
                torch.Tensor(dist.shape),
                torch.Tensor(dist.shape),
            )
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


def get_metrics_homography(
    kpts0,
    kpts1,
    image0_shape,
    image1_shape,
    H,
    thresh=3.0,
    padding=4.0,
    top_k=None,
    top_by="scores",
    return_sum=True,
    kpts_scores0=None,
    kpts_scores1=None,
):
    """
    Computes a series of metrics from the keypoints and homography

    Metrics Computed:
    - Number of keypoints
    - Number of covisible keypoints
    - Number of covisible keypoints that are correct
    - Localization error score which is an inverse of the distance between the closest
      keypoints in the two views. The score is 0 if the distance is greater than the
      threshold and 1 if the distance is 0.
    - Repeatability which is the ratio of the number of correct covisible keypoints to
      the number of covisible keypoints

    Args:
    - kpts0 (torch.Tensor): Keypoints in the first view
    - kpts1 (torch.Tensor): Keypoints in the second view
    - image0_shape (tuple): Shape of the first image
    - image1_shape (tuple): Shape of the second image
    - H (torch.Tensor): Homography matrix
    - thresh (float): Threshold for correctness
    - padding (float): Padding for the keypoints
    - top_k (int): Number of top keypoints to consider
    - top_by (str): Method to select top keypoints. Can be either "scores"
      (needs the kpts_scores{0,1}) or "dist"
    - return_sum (bool): Whether to sum the metrics for both views, or only
      for the second view
    - kpts_scores0 (torch.Tensor): Keypoint scores for the first image
    - kpts_scores1 (torch.Tensor): Keypoint scores for the second image
    """
    return_dict = {}

    if top_k is not None:
        if kpts_scores0 is None or kpts_scores1 is None:
            raise ValueError("kpts_scores0 and kpts_scores1 must be provided")

        if top_by == "scores":
            idxs0 = torch.argsort(kpts_scores0, descending=True)[:top_k]
            idxs1 = torch.argsort(kpts_scores1, descending=True)[:top_k]

            kpts0 = kpts0[idxs0]
            kpts1 = kpts1[idxs1]
        elif top_by == "dist":
            kpts0_1 = warp_points_torch(kpts0, H, inverse=False)
            kpts1_0 = warp_points_torch(kpts1, H, inverse=True)

            dists = torch.norm(kpts0_1[:, None] - kpts1[None], dim=-1)
            idxs0 = torch.argsort(dists.min(dim=1)[0], descending=False)[:top_k]
            dists = torch.norm(kpts1_0[:, None] - kpts0[None], dim=-1)
            idxs1 = torch.argsort(dists.min(dim=1)[0], descending=False)[:top_k]

            kpts0 = kpts0[idxs0]
            kpts1 = kpts1[idxs1]

            del dists

    kpts0_1 = warp_points_torch(kpts0.double(), H.double(), inverse=False)
    kpts1_0 = warp_points_torch(kpts1.double(), H.double(), inverse=True)

    metrics = compute_correctness(kpts0, kpts1, kpts0_1, kpts1_0, thresh, True)

    vis0 = torch.all(
        (kpts0_1 >= padding) & (kpts0_1 < (image1_shape - padding)),
        dim=-1,
    )
    vis1 = torch.all(
        (kpts1_0 >= padding) & (kpts1_0 < (image0_shape - padding)),
        dim=-1,
    )

    correct0 = metrics["correct0"][vis0]
    correct1 = metrics["correct1"][vis1]

    dist0 = metrics["dist0"][vis0]
    dist1 = metrics["dist1"][vis1]

    score0 = torch.max(1 - (dist0 / thresh), torch.tensor(0.0))
    score1 = torch.max(1 - (dist1 / thresh), torch.tensor(0.0))

    if return_sum:
        return_dict["num_keypoints"] = torch.tensor(kpts0.shape[0] + kpts1.shape[0])
        return_dict["num_covisible"] = vis0.sum() + vis1.sum()
        return_dict["num_covisible_correct"] = correct0.sum() + correct1.sum()
        return_dict["localization_score"] = score0.sum() + score1.sum()
        return_dict["repeatability"] = div0(
            correct0.sum() + correct1.sum(), vis0.sum() + vis1.sum()
        ).float()
    else:
        return_dict["num_keypoints"] = torch.tensor(kpts1.shape[0])
        return_dict["num_covisible"] = vis1.sum()
        return_dict["num_covisible_correct"] = correct1.sum()
        return_dict["localization_score"] = score1.sum()
        return_dict["repeatability"] = div0(correct1.sum(), vis1.sum()).float()

    return return_dict


def eval_pair_homography(
    data,
    pred,
    eval_to_0=False,
    only_eval_second=False,
    top_k=None,
    top_by="scores",
    thresh=3.0,
    padding=4.0,
):
    """
    Gets metrics per pair of images in a scene. If there are multiple transformations
    between the two images, the metrics are computed for each transformation. If the
    eval_to_0 flag is set to True, the metrics are computed with respect to the first
    transformation for both images.

    Args:
    - data: A dictionary containing the data for the pair of images
    - pred: A dictionary containing the predictions for the pair of images
    - top_k: The number of top keypoints to consider. If None, all keypoints are
      considered.
    - eval_to_0: A boolean flag that determines if the metrics are computed with respect
      to the first transformation
    - thresh: The threshold for the correctness and localization error score.

    Returns:
    - A DataFrame containing the metrics for each transformation. If eval_to_0 is True,
      the DataFrame will contain an additional column for the image.
    """
    image0 = data["view0"]["image"].permute(1, 2, 0)
    image1 = data["view1"]["image"].permute(1, 2, 0)
    H = data["H_0to1"]

    return_list = []

    def untorch(x):
        if torch.is_tensor(x):
            return x.item() if torch.numel(x) == 1 else x.numpy()
        return x

    if not isinstance(pred["keypoints0"], dict):
        metric_dict = get_metrics_homography(
            pred["keypoints0"],
            pred["keypoints1"],
            torch.tensor(image0.shape[:2][::-1]),
            torch.tensor(image1.shape[:2][::-1]),
            H,
            thresh,
            padding,
            top_k=top_k,
            top_by=top_by,
            kpts_scores0=pred["keypoint_scores0"],
            kpts_scores1=pred["keypoint_scores1"],
        )

        return_list.append(map_tensor(metric_dict, untorch))
    else:
        if eval_to_0:
            bmat0 = pred["transform0"]["0"]
            bmat1 = pred["transform1"]["0"]
            bdsize0 = pred["dsize0"]["0"]
            bdsize1 = pred["dsize1"]["0"]
            # print(pred["transform_id"]["0"])

            bkpts0 = pred["keypoints0"]["0"]
            bkpts1 = pred["keypoints1"]["0"]
            bkpts_scores0 = pred["keypoint_scores0"]["0"]
            bkpts_scores1 = pred["keypoint_scores1"]["0"]

            for i in range(len(pred["keypoints0"])):
                index = str(i)
                if not only_eval_second:

                    tmat0 = pred["transform0"][index]
                    dsize0 = pred["dsize0"][index]
                    H0 = tmat0.inverse() @ bmat0

                    kpts0 = pred["keypoints0"][index]
                    kpts_scores0 = pred["keypoint_scores0"][index]

                    metric_dict0 = get_metrics_homography(
                        bkpts0,
                        kpts0,
                        bdsize0.flip(0),
                        dsize0.flip(0),
                        H0,
                        thresh,
                        padding,
                        return_sum=False,
                        top_k=top_k,
                        top_by=top_by,
                        kpts_scores0=bkpts_scores0,
                        kpts_scores1=kpts_scores0,
                    )
                    metric_dict0["transform"] = pred["transform_id"][index]
                    metric_dict0["image"] = torch.tensor(0)
                    return_list.append(map_tensor(metric_dict0, untorch))

                tmat1 = pred["transform1"][index]
                dsize1 = pred["dsize1"][index]

                H1 = tmat1.inverse() @ bmat1

                kpts1 = pred["keypoints1"][index]
                kpts_scores1 = pred["keypoint_scores1"][index]

                metric_dict1 = get_metrics_homography(
                    bkpts1,
                    kpts1,
                    bdsize1.flip(0),
                    dsize1.flip(0),
                    H1,
                    thresh,
                    padding,
                    return_sum=False,
                    top_k=top_k,
                    top_by=top_by,
                    kpts_scores0=bkpts_scores1,
                    kpts_scores1=kpts_scores1,
                )
                metric_dict1["transform"] = pred["transform_id"][index]
                metric_dict1["image"] = torch.tensor(1)
                return_list.append(map_tensor(metric_dict1, untorch))
        else:
            for i in range(len(pred["keypoints0"])):
                index = str(i)
                tmat0 = pred["transform0"][index]
                dsize0 = pred["dsize0"][index]
                tmat1 = pred["transform1"][index]
                dsize1 = pred["dsize1"][index]

                H_new = tmat1 @ H @ tmat0.inverse()

                kpts0 = pred["keypoints0"][index]
                kpts1 = pred["keypoints1"][index]
                kpts_scores0 = pred["keypoint_scores0"][index]
                kpts_scores1 = pred["keypoint_scores1"][index]

                metric_dict = get_metrics_homography(
                    kpts0,
                    kpts1,
                    dsize0.flip(0),
                    dsize1.flip(0),
                    H_new,
                    thresh,
                    padding,
                    top_k=top_k,
                    top_by=top_by,
                    kpts_scores0=kpts_scores0,
                    kpts_scores1=kpts_scores1,
                )
                metric_dict["transform"] = pred["transform_id"][index]
                return_list.append(map_tensor(metric_dict, untorch))

    pair_df = pd.DataFrame.from_records(return_list)
    pair_df["name"] = data["name"][0]

    return pair_df


def get_metrics_depth(
    kpts0,
    kpts1,
    data_,
    thresh=3.0,
    padding=4.0,
    top_k=None,
    top_by="scores",
    return_sum=True,
    kpts_scores0=None,
    kpts_scores1=None,
):
    return_dict = {}

    depth0 = data_["view0"]["depth"]
    depth1 = data_["view1"]["depth"]
    cam0 = data_["view0"]["camera"]
    cam1 = data_["view1"]["camera"]
    T0_1 = data_["T_0to1"]
    T1_0 = T0_1.inv()

    if top_k is not None:
        if kpts_scores0 is None or kpts_scores1 is None:
            raise ValueError("kpts_scores0 and kpts_scores1 must be provided")

        if top_by == "scores":
            idxs0 = torch.argsort(kpts_scores0, descending=True)[:top_k]
            idxs1 = torch.argsort(kpts_scores1, descending=True)[:top_k]

            kpts0 = kpts0[idxs0]
            kpts1 = kpts1[idxs1]
        elif top_by == "dist":
            d0, valid0 = sample_depth(kpts0[None], depth0[None])
            d1, valid1 = sample_depth(kpts1[None], depth1[None])

            kpts0_1, _ = project(
                kpts0[None], d0, depth1[None], cam0, cam1, T0_1, valid0, 3.0
            )  # [B, M, 2]
            kpts1_0, _ = project(
                kpts1[None], d1, depth0[None], cam1, cam0, T1_0, valid1, 3.0
            )  # [B, N, 2]
            dists = torch.norm(kpts0_1[:, None] - kpts1[None, :, None], dim=-1)
            values, indices = torch.sort(dists.min(dim=1)[0], descending=False)
            idxs0 = indices[~values.isnan()[:]][:top_k]
            dists = torch.norm(kpts1_0[:, None] - kpts0[None, :, None], dim=-1)
            values, indices = torch.sort(dists.min(dim=1)[0], descending=False)
            idxs1 = indices[~values.isnan()[:]][:top_k]

            kpts0 = kpts0[idxs0].squeeze(0)
            kpts1 = kpts1[idxs1].squeeze(0)

            del dists

    d0, valid0 = sample_depth(kpts0[None], depth0[None])
    d1, valid1 = sample_depth(kpts1[None], depth1[None])

    kpts0_1, vis0 = project(
        kpts0[None], d0, depth1[None], cam0, cam1, T0_1, valid0, 3.0
    )  # [M, 2]
    kpts1_0, vis1 = project(
        kpts1[None], d1, depth0[None], cam1, cam0, T1_0, valid1, 3.0
    )  # [N, 2]

    kpts0 = kpts0.squeeze(0)
    kpts1 = kpts1.squeeze(0)
    kpts0_1 = kpts0_1.squeeze(0)
    kpts1_0 = kpts1_0.squeeze(0)
    vis0 = vis0.squeeze(0)
    vis1 = vis1.squeeze(0)

    metrics = compute_correctness(kpts0, kpts1, kpts0_1, kpts1_0, thresh, True)

    correct0 = metrics["correct0"][vis0]
    correct1 = metrics["correct1"][vis1]

    dist0 = metrics["dist0"][vis0]
    dist1 = metrics["dist1"][vis1]

    score0 = torch.max(1 - (dist0 / thresh), torch.tensor(0.0))
    score1 = torch.max(1 - (dist1 / thresh), torch.tensor(0.0))

    if return_sum:
        return_dict["num_keypoints"] = torch.tensor(kpts0.shape[0] + kpts1.shape[0])
        return_dict["num_covisible"] = vis0.sum() + vis1.sum()
        return_dict["num_covisible_correct"] = correct0.sum() + correct1.sum()
        return_dict["localization_score"] = score0.sum() + score1.sum()
        # return_dict["repeatability"] = div0(
        #     correct0.sum() + correct1.sum(), vis0.sum() + vis1.sum()
        # ).float()
    else:
        return_dict["num_keypoints"] = torch.tensor(kpts1.shape[0])
        return_dict["num_covisible"] = vis1.sum()
        return_dict["num_covisible_correct"] = correct1.sum()
        return_dict["localization_score"] = score1.sum()
        # return_dict["repeatability"] = div0(correct1.sum(), vis1.sum()).float()

    return return_dict


def eval_pair_depth(
    data,
    pred,
    eval_to_0=False,
    only_eval_second=False,
    top_k=None,
    top_by="scores",
    thresh=3.0,
    padding=4.0,
):
    return_list = []

    def untorch(x):
        if torch.is_tensor(x):
            return x.item() if torch.numel(x) == 1 else x.numpy()
        return x

    if not isinstance(pred["keypoints0"], dict):
        metric_dict = get_metrics_depth(
            pred["keypoints0"],
            pred["keypoints1"],
            data,
            thresh,
            padding,
            top_k=top_k,
            top_by=top_by,
            kpts_scores0=pred["keypoint_scores0"],
            kpts_scores1=pred["keypoint_scores1"],
        )

        return_list.append(map_tensor(metric_dict, untorch))
    else:
        raise NotImplementedError("Evaluation of transformed images not supported yet.")
        # if eval_to_0:
        #     bmat0 = pred["transform0"]["0"]
        #     bmat1 = pred["transform1"]["0"]
        #     bdsize0 = pred["dsize0"]["0"]
        #     bdsize1 = pred["dsize1"]["0"]
        #     # print(pred["transform_id"]["0"])

        #     bkpts0 = pred["keypoints0"]["0"]
        #     bkpts1 = pred["keypoints1"]["0"]
        #     bkpts_scores0 = pred["keypoint_scores0"]["0"]
        #     bkpts_scores1 = pred["keypoint_scores1"]["0"]

        #     for i in range(len(pred["keypoints0"])):
        #         index = str(i)
        #         if not only_eval_second:

        #             tmat0 = pred["transform0"][index]
        #             dsize0 = pred["dsize0"][index]
        #             H0 = tmat0.inverse() @ bmat0

        #             kpts0 = pred["keypoints0"][index]
        #             kpts_scores0 = pred["keypoint_scores0"][index]

        #             metric_dict0 = get_metrics_depth(
        #                 bkpts0,
        #                 kpts0,
        #                 bdsize0.flip(0),
        #                 dsize0.flip(0),
        #                 H0,
        #                 thresh,
        #                 padding,
        #                 return_sum=False,
        #                 top_k=top_k,
        #                 top_by=top_by,
        #                 kpts_scores0=bkpts_scores0,
        #                 kpts_scores1=kpts_scores0,
        #             )
        #             metric_dict0["transform"] = pred["transform_id"][index]
        #             metric_dict0["image"] = torch.tensor(0)
        #             return_list.append(map_tensor(metric_dict0, untorch))

        #         tmat1 = pred["transform1"][index]
        #         dsize1 = pred["dsize1"][index]

        #         H1 = tmat1.inverse() @ bmat1

        #         kpts1 = pred["keypoints1"][index]
        #         kpts_scores1 = pred["keypoint_scores1"][index]

        #         metric_dict1 = get_metrics_depth(
        #             bkpts1,
        #             kpts1,
        #             bdsize1.flip(0),
        #             dsize1.flip(0),
        #             H1,
        #             thresh,
        #             padding,
        #             return_sum=False,
        #             top_k=top_k,
        #             top_by=top_by,
        #             kpts_scores0=bkpts_scores1,
        #             kpts_scores1=kpts_scores1,
        #         )
        #         metric_dict1["transform"] = pred["transform_id"][index]
        #         metric_dict1["image"] = torch.tensor(1)
        #         return_list.append(map_tensor(metric_dict1, untorch))
        # else:
        #     for i in range(len(pred["keypoints0"])):
        #         index = str(i)
        #         tmat0 = pred["transform0"][index]
        #         dsize0 = pred["dsize0"][index]
        #         tmat1 = pred["transform1"][index]
        #         dsize1 = pred["dsize1"][index]

        #         H_new = tmat1 @ H @ tmat0.inverse()

        #         kpts0 = pred["keypoints0"][index]
        #         kpts1 = pred["keypoints1"][index]
        #         kpts_scores0 = pred["keypoint_scores0"][index]
        #         kpts_scores1 = pred["keypoint_scores1"][index]

        #         metric_dict = get_metrics_depth(
        #             kpts0,
        #             kpts1,
        #             dsize0.flip(0),
        #             dsize1.flip(0),
        #             H_new,
        #             thresh,
        #             padding,
        #             top_k=top_k,
        #             top_by=top_by,
        #             kpts_scores0=kpts_scores0,
        #             kpts_scores1=kpts_scores1,
        #         )
        #         metric_dict["transform"] = pred["transform_id"][index]
        #         return_list.append(map_tensor(metric_dict, untorch))

    pair_df = pd.DataFrame.from_records(return_list)
    pair_df["name"] = data["name"][0]

    return pair_df


def get_homography_matches(kpts0, kpts1, H, thresh=3.0):
    kpts0_1 = warp_points_torch(kpts0, H, inverse=False)
    kpts1_0 = warp_points_torch(kpts1, H, inverse=True)

    dist0 = torch.norm(kpts0_1[:, :, None, :] - kpts1[:, None, :, :], dim=-1)
    dist1 = torch.norm(kpts0[:, :, None, :] - kpts1_0[:, None, :, :], dim=-1)
    dist = torch.max(dist0, dist1)

    inf = dist.new_tensor(float("inf"))
    dist = torch.where(dist < thresh, dist, inf)

    min0 = dist.min(-1).indices
    min1 = dist.min(-2).indices
    ismin0 = torch.zeros(dist.shape, dtype=torch.bool, device=dist.device)
    ismin1 = ismin0.clone()

    ismin0.scatter_(-1, min0.unsqueeze(-1), value=1)
    ismin1.scatter_(-2, min1.unsqueeze(-2), value=1)
    positive = ismin0 & ismin1

    pts0 = kpts0[positive.any(-1)]
    pts1 = kpts1[positive.any(-2)]
    unmatched0 = min0.new_tensor(-1)

    m0 = torch.where(positive.any(-1), min0, unmatched0)

    validm0 = m0 != -1

    pts0 = kpts0[:, torch.arange(m0.shape[1], device=m0.device)[validm0[0]], :]
    pts1 = kpts1[:, m0[0][validm0[0]], :]

    return pts0, pts1


def get_depth_matches(kpts0, kpts1, depth0, depth1, camera0, camera1, T_0to1):
    d0, valid0 = sample_depth(kpts0, depth0)
    d1, valid1 = sample_depth(kpts1, depth1)

    kpts0_1, visible0 = project(
        kpts0, d0, depth1, camera0, camera1, T_0to1, valid0, 1.0
    )  # [B, M, 2]
    kpts1_0, visible1 = project(
        kpts1, d1, depth0, camera1, camera0, T_0to1, valid1, 1.0
    )  # [B, N, 2]
    mask_visible = visible0.unsqueeze(-1) & visible1.unsqueeze(-2)

    dist0 = torch.norm(kpts0_1[:, :, None, :] - kpts1[:, None, :, :], dim=-1)
    dist1 = torch.norm(kpts0[:, :, None, :] - kpts1_0[:, None, :, :], dim=-1)
    dist = torch.max(dist0, dist1)
    inf = dist.new_tensor(float("inf"))
    dist = torch.where(mask_visible, dist, inf)

    min0 = dist.min(-1).indices
    min1 = dist.min(-2).indices
    ismin0 = torch.zeros(dist.shape, dtype=torch.bool, device=dist.device)
    ismin1 = ismin0.clone()

    ismin0.scatter_(-1, min0.unsqueeze(-1), value=1)
    ismin1.scatter_(-2, min1.unsqueeze(-2), value=1)
    positive = ismin0 & ismin1

    pts0 = kpts0[positive.any(-1)]
    pts1 = kpts1[positive.any(-2)]
    unmatched0 = min0.new_tensor(-1)

    m0 = torch.where(positive.any(-1), min0, unmatched0)

    validm0 = m0 != -1

    pts0 = kpts0[:, torch.arange(m0.shape[1], device=m0.device)[validm0[0]], :]
    pts1 = kpts1[:, m0[0][validm0[0]], :]

    return pts0, pts1


def get_desc_matches(kpts0, kpts1, desc0, desc1):
    if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:
        return torch.Tensor(kpts0.shape), torch.Tensor(kpts1.shape)

    distances = distance_matrix(
        desc0,
        desc1,
    ).to(desc0.device)
    if distances.dim() == 2:
        distances = distances.unsqueeze(0)

    n_amin = torch.argmin(distances, dim=-1)
    m_amin = torch.argmin(distances, dim=-2)

    nnnn = m_amin.gather(1, n_amin)
    n_ix = torch.arange(distances.shape[-2], device=distances.device)
    mask = nnnn[0] == n_ix

    pts0 = kpts0[:, torch.nonzero(mask, as_tuple=False)[:, 0], :]
    pts1 = kpts1[:, n_amin[0][mask], :]

    return pts0, pts1


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
    check_keys_recursive(data, ["view0", "view1"])
    check_keys_recursive(pred, ["keypoints0", "keypoints1"])

    T_gt = data["T_0to1"].unsqueeze(0)
    kp0 = pred["keypoints0"].unsqueeze(0)
    kp1 = pred["keypoints1"].unsqueeze(0)
    if conf["use_gt"]:
        camera0 = data["view0"]["camera"].unsqueeze(0)
        camera1 = data["view1"]["camera"].unsqueeze(0)
        depth0 = data["view0"]["depth"].unsqueeze(0)
        depth1 = data["view1"]["depth"].unsqueeze(0)
        pts0, pts1 = get_depth_matches(kp0, kp1, depth0, depth1, camera0, camera1, T_gt)
    else:
        descriptors0 = pred["descriptors0"].unsqueeze(0)
        descriptors1 = pred["descriptors1"].unsqueeze(0)
        pts0, pts1 = get_desc_matches(kp0, kp1, descriptors0, descriptors1)

    results = {}

    estimator = load_estimator("relative_pose", conf["estimator"])(conf)
    data_ = {
        "m_kpts0": pts0,
        "m_kpts1": pts1,
        "camera0": data["view0"]["camera"],
        "camera1": data["view1"]["camera"],
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

    return pd.DataFrame.from_records([results])


def eval_homography_robust(data, pred, conf):
    check_keys_recursive(data, ["view0", "view1", "H_0to1"])
    check_keys_recursive(pred, ["keypoints0", "keypoints1"])

    H_gt = data["H_0to1"].unsqueeze(0)
    kp0 = pred["keypoints0"].unsqueeze(0)
    kp1 = pred["keypoints1"].unsqueeze(0)
    if conf["use_gt"]:
        pts0, pts1 = get_homography_matches(kp0, kp1, H_gt, conf["correctness_threshold"])
    else:
        descriptors0 = pred["descriptors0"].unsqueeze(0)
        descriptors1 = pred["descriptors1"].unsqueeze(0)
        pts0, pts1 = get_desc_matches(kp0, kp1, descriptors0, descriptors1)

    results = {}

    estimator = load_estimator("homography", conf["estimator"])(conf)

    data_ = {
        "m_kpts0": pts0,
        "m_kpts1": pts1,
    }

    est = estimator(data_)

    if not est["success"]:
        results["H_error_ransac"] = float("inf")
        results["ransac_inl"] = 0
        results["ransac_inl%"] = 0
    else:
        # R, t, inl = ret
        M = est["M_0to1"]
        inl = est["inliers"].numpy()
        results["H_error_ransac"] = homography_corner_error(M, H_gt, data["view0"]["image_size"]).item()
        results["ransac_inl"] = np.sum(inl)
        results["ransac_inl%"] = np.mean(inl)

    return pd.DataFrame.from_records([results])


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


def calc_auc():
    pass


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
