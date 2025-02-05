from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import seaborn as sns
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..datasets import get_dataset
from ..models.cache_loader import CacheLoader
from ..settings import EVAL_PATH
from ..utils.export_predictions import export_predictions
from ..utils.tensor import map_tensor
from ..utils.tools import AUCMetric
from .eval_pipeline import EvalPipeline
from .io import get_eval_parser, load_model, parse_eval_args
from .utils import eval_homography_robust, eval_pair_homography
from ..geometry.homography import warp_points_torch


def plot_scene_summary(summaries, columns, title, ylabel):
    fig, axes = plt.subplots(2, 1, figsize=(16, 14), sharey=True)

    illum = summaries[summaries["scene"].map(lambda x: x.startswith("i"))]
    view = summaries[summaries["scene"].map(lambda x: x.startswith("v"))]

    if not isinstance(columns, list):
        columns = [columns]

    illum = illum.melt(
        id_vars=["scene"], value_vars=columns, var_name="key", value_name="value"
    )
    view = view.melt(
        id_vars=["scene"], value_vars=columns, var_name="key", value_name="value"
    )
    columns = "value"
    hue = "key"

    sns.lineplot(
        data=illum,
        x="scene",
        y=columns,
        hue=hue,
        markers=True,
        dashes=False,
        ax=axes[0],
    )
    sns.lineplot(
        data=view,
        x="scene",
        y=columns,
        hue=hue,
        markers=True,
        dashes=False,
        ax=axes[1],
    )

    if len(columns) == 1:
        axes[0].axhline(
            y=illum.mean(axis=0, numeric_only=True)[columns[0]],
            label="illumination mean",
        )
        axes[0].axhline(
            y=summaries.mean(axis=0, numeric_only=True)[columns[0]],
            color="black",
            label="global mean",
        )
        axes[1].axhline(
            y=view.mean(axis=0, numeric_only=True)[columns[0]], label="viewpoint mean"
        )
        axes[1].axhline(
            y=summaries.mean(axis=0, numeric_only=True)[columns[0]],
            color="black",
            label="global mean",
        )

    axes[0].legend()
    axes[1].legend()

    axes[0].set_title(f"{title} per scene (illumination)")
    axes[0].set_xlabel("Scenes")
    axes[0].set_ylabel(ylabel)

    axes[1].set_title(f"{title} per scene (viewpoint)")
    axes[1].set_xlabel("Scenes")
    axes[1].set_ylabel(ylabel)

    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_rotation(60)
            label.set_ha("right")

    fig.tight_layout()

    return fig


class HPatchesPipeline(EvalPipeline):
    default_conf = {
        "data": {
            "batch_size": 1,
            "name": "hpatches",
            "num_workers": 16,
            "preprocessing": {
                "resize": 480,  # we also resize during eval to have comparable metrics
                "side": "short",
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
            "summarize_by_scene": True,
            "ransac_th": 1.0,
        },
    }
    export_keys = [
        "keypoints0",
        "keypoints1",
        "keypoint_scores0",
        "keypoint_scores1",
    ]

    optional_export_keys = [
        "heatmap0",
        "heatmap1",
        "descriptors0",
        "descriptors1",
    ]

    def _init(self, conf):
        pass

    @classmethod
    def get_dataloader(self, data_conf=None):
        data_conf = data_conf if data_conf else self.default_conf["data"]
        dataset = get_dataset("hpatches")(data_conf)
        return dataset.get_data_loader("test")

    def get_predictions(
        self, experiment_dir, model=None, overwrite=False, get_last=False
    ):
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(
                    self.conf.model, self.conf.checkpoint, get_last=get_last
                )
            export_predictions(
                self.get_dataloader(self.conf.data),
                model,
                pred_file,
                keys=self.export_keys,
                optional_keys=self.optional_export_keys,
            )
        return pred_file

    def run_eval(self, loader, pred_file):
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
        for data in tqdm(loader):
            pred = cache_loader(data)
            data = map_tensor(data, lambda x: torch.squeeze(x, dim=0))
            scene_name = str(Path(data["name"][0]).parent)
            for top_k in conf.top_k_thresholds:
                for top_by in conf.top_k_by:
                    kpts0 = pred["keypoints0"]
                    kpts1 = pred["keypoints1"]
                    kpts_score0 = pred["keypoint_scores0"]
                    kpts_score1 = pred["keypoint_scores1"]
                    H = data["H_0to1"]
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
                        if "descriptors0" in pred:
                            new_pred["descriptors0"] = pred["descriptors0"][idxs0]
                            new_pred["descriptors1"] = pred["descriptors1"][idxs1]
                    elif top_by == "dist":
                        kpts0_1 = warp_points_torch(kpts0, H, inverse=False)
                        kpts1_0 = warp_points_torch(kpts1, H, inverse=True)

                        dists = torch.norm(kpts0_1[:, None] - kpts1[None], dim=-1)
                        idxs0 = torch.argsort(dists.min(dim=1)[0], descending=False)[
                            :top_k
                        ]
                        dists = torch.norm(kpts1_0[:, None] - kpts0[None], dim=-1)
                        idxs1 = torch.argsort(dists.min(dim=1)[0], descending=False)[
                            :top_k
                        ]

                        new_pred = {
                            "keypoints0": kpts0[idxs0],
                            "keypoints1": kpts1[idxs1],
                            "keypoint_scores0": kpts_score0[idxs0],
                            "keypoint_scores1": kpts_score1[idxs1],
                        }
                        if "descriptors0" in pred:
                            new_pred["descriptors0"] = pred["descriptors0"][idxs0]
                            new_pred["descriptors1"] = pred["descriptors1"][idxs1]

                        del dists

                    pair_metrics = eval_pair_homography(
                        data,
                        new_pred,
                        eval_to_0=False,
                        thresh=conf.correctness_threshold,
                        padding=conf.padding,
                    )
                    # This is a quick fix. Don't do this.
                    # Handle edge case when calculating repeatability
                    pair_metrics["top_k"] = top_k
                    pair_metrics["top_by"] = str(top_by)
                    pair_metrics["scene"] = str(scene_name)
                    pair_metrics["name"] = data["name"][0]
                    if self.conf.eval.estimator:
                        for th in test_thresholds:
                            pose_metrics = eval_homography_robust(
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
        #                 self.conf.model.extractor.max_num_keypoints
        #                 if self.conf.model.extractor.max_num_keypoints is not None
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
            if "H_error" in group.columns:
                mAA, aucs = calc_pose_metrics(group["H_error"])
                aggregations["H_error_mAA"] = mAA
                aggregations["H_error@1px"] = aucs[0]
                aggregations["H_error@3px"] = aucs[1]
                aggregations["H_error@5px"] = aucs[2]
            return pd.Series(aggregations)

        groupby_columns = ["top_k", "top_by"]

        if self.conf.eval.summarize_by_scene:
            groupby_columns.append("scene")

        if self.conf.eval.estimator:
            groupby_columns.append("ransac_th")

        # Perform the aggregation
        summaries = (
            results.groupby(groupby_columns).apply(custom_aggregation).reset_index()
        )

        figures = {}
        if "num_covisible_correct" in summaries.columns:
            figures["ncc_ratio"] = plot_scene_summary(
                summaries,
                "num_covisible_correct",
                title="Number of correct covisible points",
                ylabel="NCC Ratio",
            )

        if "localization_score" in summaries.columns:
            figures["loc_scores"] = plot_scene_summary(
                summaries,
                "localization_score",
                title="Sum of localization scores",
                ylabel="Localization Score",
            )

        if "repeatability" in summaries.columns:
            figures["repeatability"] = plot_scene_summary(
                summaries,
                "repeatability",
                title="Repeatability",
                ylabel="Repeatability",
            )

        if self.conf.eval.estimator:
            figures["H_error"] = plot_scene_summary(
                summaries,
                ["H_error@1px", "H_error@3px", "H_error@5px", "H_error_mAA"],
                title="Homography Error",
                ylabel="Homography Error",
            )

        return summaries, figures, results


if __name__ == "__main__":
    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(HPatchesPipeline.default_conf)

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

    pipeline = HPatchesPipeline(conf)
    s, f, r = pipeline.run(
        experiment_dir,
        overwrite=args.overwrite,
        overwrite_eval=args.overwrite_eval,
        get_last=args.get_last,
    )

    # print results
    pprint(s)
    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()
