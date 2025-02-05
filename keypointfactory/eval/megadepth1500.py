import logging
import zipfile
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..datasets import get_dataset
from ..models.cache_loader import CacheLoader
from ..settings import DATA_PATH, EVAL_PATH
from ..utils.export_predictions import export_predictions
from ..utils.tensor import map_tensor
from ..utils.tools import AUCMetric
from .eval_pipeline import EvalPipeline
from .io import get_eval_parser, load_model, parse_eval_args
from .utils import eval_pair_depth, eval_relative_pose_robust
from ..utils.tools import AUCMetric
from ..geometry.depth import project, sample_depth

logger = logging.getLogger(__name__)


class MegaDepth1500Pipeline(EvalPipeline):
    default_conf = {
        "data": {
            "name": "image_pairs",
            "pairs": "megadepth1500/pairs_calibrated.txt",
            "root": "megadepth1500/",
            "extra_data": "relative_pose",
            "preprocessing": {
                "side": "long",
            },
        },
        "model": {
            "ground_truth": {
                "name": None,  # remove gt matches
            }
        },
        "eval": {
            "correctness_threshold": 3.0,
            "padding": 4.0,
            "top_k_thresholds": None,  # None means all keypoints, otherwise a list of thresholds (which can also include None) # noqa: E501
            "top_k_by": "scores",  # either "scores" or "distances", or list of both
            "use_gt": False,
        },
    }

    export_keys = [
        "keypoints0",
        "keypoints1",
        "keypoint_scores0",
        "keypoint_scores1",
    ]
    optional_export_keys = []

    def _init(self, conf):
        if not (DATA_PATH / "megadepth1500").exists():
            logger.info("Downloading the MegaDepth-1500 dataset.")
            url = "https://cvg-data.inf.ethz.ch/megadepth/megadepth1500.zip"
            zip_path = DATA_PATH / url.rsplit("/", 1)[-1]
            zip_path.parent.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(url, zip_path)
            with zipfile.ZipFile(zip_path) as fid:
                fid.extractall(DATA_PATH)
            zip_path.unlink()

    @classmethod
    def get_dataloader(self, data_conf=None):
        """Returns a data loader with samples for each eval datapoint"""
        data_conf = data_conf if data_conf else self.default_conf["data"]
        dataset = get_dataset(data_conf["name"])(data_conf)
        return dataset.get_data_loader("test")

    def get_predictions(
        self, experiment_dir, model=None, overwrite=False, get_last=False
    ):
        """Export a prediction file for each eval datapoint"""
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(self.conf.model, self.conf.checkpoint, get_last)
            export_predictions(
                self.get_dataloader(self.conf.data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
            )
        return pred_file

    def run_eval(self, loader, pred_file):
        """Run the eval on cached predictions"""
        assert pred_file.exists()
        results = defaultdict(list)
        conf = self.conf.eval

        if isinstance(conf.top_k_thresholds, int) or conf.top_k_thresholds is None:
            conf.top_k_thresholds = [conf.top_k_thresholds]
        if isinstance(conf.top_k_by, str) or conf.top_k_by is None:
            conf.top_k_by = [conf.top_k_by]

        test_thresholds = (
            ([conf.ransac_th] if conf.ransac_th > 0 else [0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
            if not isinstance(conf.ransac_th, Iterable)
            else conf.ransac_th
        )

        df_list = []
        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()
        for i, data in enumerate(tqdm(loader)):
            assert "depth" in data["view0"]
            pred = cache_loader(data)
            data = map_tensor(data, lambda x: torch.squeeze(x, dim=0))
            scene_name = data["name"][0].split("-")[0]
            for top_k in conf.top_k_thresholds:
                for top_by in conf.top_k_by:
                    kpts0 = pred["keypoints0"]
                    kpts1 = pred["keypoints1"]
                    kpts_score0 = pred["keypoint_scores0"]
                    kpts_score1 = pred["keypoint_scores1"]
                    new_pred = {}
                    if top_by == "scores":
                        idxs0 = torch.argsort(kpts_score0, descending=True)[:top_k]
                        idxs1 = torch.argsort(kpts_score1, descending=True)[:top_k]

                        new_pred = {
                            "keypoints0": kpts0[idxs0],
                            "keypoints1": kpts1[idxs1],
                            "keypoint_scores0": kpts_score0[idxs0],
                            "keypoint_scores1": kpts_score1[idxs1],
                        }
                    elif top_by == "dist":
                        depth0 = data["view0"]["depth"]
                        depth1 = data["view1"]["depth"]
                        cam0 = data["view0"]["camera"]
                        cam1 = data["view1"]["camera"]
                        T0_1 = data["T_0to1"]
                        T1_0 = data["T_1to0"]

                        d0, valid0 = sample_depth(kpts0[None], depth0[None])
                        d1, valid1 = sample_depth(kpts1[None], depth1[None])

                        kpts0_1, _ = project(
                            kpts0[None], d0, depth1[None], cam0, cam1, T0_1, valid0, 3.0
                        )  # [B, M, 2]
                        kpts1_0, _ = project(
                            kpts1[None], d1, depth0[None], cam1, cam0, T1_0, valid1, 3.0
                        )  # [B, N, 2]
                        dists = torch.norm(
                            kpts0_1[:, None] - kpts1[None, :, None], dim=-1
                        )
                        values, indices = torch.sort(
                            dists.min(dim=1)[0], descending=False
                        )
                        idxs0 = indices[~values.isnan()[:]][:top_k]
                        dists = torch.norm(
                            kpts1_0[:, None] - kpts0[None, :, None], dim=-1
                        )
                        values, indices = torch.sort(
                            dists.min(dim=1)[0], descending=False
                        )
                        idxs1 = indices[~values.isnan()[:]][:top_k]

                        new_pred = {
                            "keypoints0": kpts0[idxs0],
                            "keypoints1": kpts1[idxs1],
                            "keypoint_scores0": kpts_score0[idxs0],
                            "keypoint_scores1": kpts_score1[idxs1],
                        }

                        del dists

                    pair_metrics = eval_pair_depth(
                        data,
                        new_pred,
                        thresh=conf.correctness_threshold,
                        padding=conf.padding,
                    )
                    pair_metrics["top_k"] = top_k
                    pair_metrics["top_by"] = top_by
                    pair_metrics["scene"] = scene_name
                    pair_metrics["name"] = data["name"][0]
                    if self.conf.eval.estimator:
                        for th in test_thresholds:
                            pose_metrics = eval_relative_pose_robust(
                                data, new_pred, {**self.conf.eval, "ransac_th": th}
                            )

                            pair_metrics = pair_metrics.join(pose_metrics, how="left")
                            pair_metrics["ransac_th"] = th

                            df_list.append(pair_metrics)
                    else:
                        df_list.append(pair_metrics)

        results = pd.concat(df_list)

        # results["repeatability"] = results["num_covisible_correct"] / (
        #     2
        #     * results["top_k"].map(
        #         lambda x: (
        #             x
        #             if x is not None
        #             else (
        #                 self.conf.model.max_num_keypoints
        #                 if self.conf.model.max_num_keypoints is not None
        #                 else float("inf")
        #             )
        #         )
        #     )
        # )  # Multiple top_k by 2 because we take
        # # sum of correct points from two images

        results["repeatability"] = (
            results["num_covisible_correct"] / results["num_covisible"]
        )
        results["localization"] = (
            results["localization_score"] / results["num_covisible_correct"]
        )

        def calc_pose_metrics(df):
            aucs = AUCMetric([1, 3, 5], elements=df, return_mean=False).compute()
            if not isinstance(aucs, list):
                aucs = [aucs] * 3
            return np.nanmean(aucs), aucs

        def custom_aggregation(group):
            aggregations = {
                "num_keypoints": group["num_keypoints"].sum(),
                "num_covisible": group["num_covisible"].sum(),
                "num_covisible_correct": group["num_covisible_correct"].sum(),
                "localization_score": group["localization_score"].sum(),
                "repeatability": group["repeatability"].mean(),
                "localization": group["localization"].mean(),
            }
            if "rel_pose_error" in group.columns:
                mAA, aucs = calc_pose_metrics(group["rel_pose_error"])
                aggregations["rel_pose_error_mAA"] = mAA
                aggregations["rel_pose_error@1px"] = aucs[0]
                aggregations["rel_pose_error@3px"] = aucs[1]
                aggregations["rel_pose_error@5px"] = aucs[2]
            return pd.Series(aggregations)

        groupby_columns = ["top_k", "top_by", "scene"]

        if self.conf.eval.estimator:
            groupby_columns.append("ransac_th")

        # Perform the aggregation
        summaries = (
            results.groupby(groupby_columns).apply(custom_aggregation).reset_index()
        )

        return summaries, {}, results


if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(MegaDepth1500Pipeline.default_conf)

    # mingle paths
    output_dir = Path(EVAL_PATH, dataset_name)
    output_dir.mkdir(exist_ok=True, parents=True)

    name, conf = parse_eval_args(
        dataset_name,
        args,
        "configs/",
        default_conf,
    )

    experiment_dir = output_dir / name
    experiment_dir.mkdir(exist_ok=True)

    pipeline = MegaDepth1500Pipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir,
        overwrite=args.overwrite,
        overwrite_eval=args.overwrite_eval,
    )

    pprint(s)

    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()
