from collections import defaultdict
from pathlib import Path
from pprint import pprint
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..datasets import get_dataset
from ..models.cache_loader import CacheLoader
from ..settings import EVAL_PATH
from ..utils.export_predictions import export_predictions
from ..utils.tensor import map_tensor
from .eval_pipeline import EvalPipeline
from .io import get_eval_parser, load_model, parse_eval_args
from .utils import eval_pair_homography, eval_homography_robust
from ..utils.tools import AUCMetric


def get_hpatches_scenes(data_loader, cache_loader, squeezed=True):
    name = None
    group = []

    for i, data in enumerate(data_loader):
        if name is None:
            name = str(Path(data["name"][0]).parent)
            if squeezed:
                group.append(
                    (
                        map_tensor(data, lambda x: torch.squeeze(x, dim=0)),
                        cache_loader(data),
                    )
                )
            else:
                group.append((data, cache_loader(data)))
            continue

        if str(Path(data["name"][0]).parent) == name:
            if squeezed:
                group.append(
                    (
                        map_tensor(data, lambda x: torch.squeeze(x, dim=0)),
                        cache_loader(data),
                    )
                )
            else:
                group.append((data, cache_loader(data)))
        else:
            yield name, group
            name = str(Path(data["name"][0]).parent)
            if squeezed:
                group = [
                    (
                        map_tensor(data, lambda x: torch.squeeze(x, dim=0)),
                        cache_loader(data),
                    )
                ]
            else:
                group = [(data, cache_loader(data))]

    if len(group) > 0:
        yield name, group


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
        },
    }
    export_keys = [
        "keypoints0",
        "keypoints1",
        "keypoint_scores0",
        "keypoint_scores1",
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

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        pred_file = experiment_dir / "predictions.h5"
        if not pred_file.exists() or overwrite:
            if model is None:
                model = load_model(self.conf.model, self.conf.checkpoint)
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
                    pair_metrics = eval_pair_homography(
                        data,
                        pred,
                        eval_to_0=False,
                        top_k=int(top_k) if top_k is not None else top_k,
                        top_by=top_by,
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
                                data, pred, {**self.conf.eval, "ransac_th": th}
                            )

                            pair_metrics = pair_metrics.join(pose_metrics, how="left")
                            pair_metrics["ransac_th"] = th

                            df_list.append(pair_metrics)
                    else:
                        df_list.append(pair_metrics)

        results = pd.concat(df_list)

        results["repeatability"] = results["num_covisible_correct"] / (
            2
            * results["top_k"].map(
                lambda x: (
                    x
                    if x is not None
                    else (
                        self.conf.model.extractor.max_num_keypoints
                        if self.conf.model.extractor.max_num_keypoints is not None
                        else float("inf")
                    )
                )
            )
        )  # Multiple top_k by 2 because we take
        # sum of correct points from two images

        def calc_auc(df):
            auc = AUCMetric(list(range(1, 11)), df)
            return auc.compute()

        groupby_columns = ["top_k", "top_by"]
        if self.conf.eval.summarize_by_scene:
            groupby_columns.append("scene")
        agg_funcs = {
            "num_keypoints": ("num_keypoints", "sum"),
            "num_covisible": ("num_covisible", "sum"),
            "num_covisible_correct": ("num_covisible_correct", "sum"),
            "localization_score": ("localization_score", "sum"),
            "repeatability": ("repeatability", "mean"),
        }

        if self.conf.eval.estimator:
            agg_funcs["H_error_auc"] = ("H_error_ransac", calc_auc)
            groupby_columns.append("ransac_th")

        # Perform the aggregation
        summaries = (
            results.groupby(groupby_columns)
            .agg(
                **{
                    key: pd.NamedAgg(column=value[0], aggfunc=value[1])
                    for key, value in agg_funcs.items()
                }
            )
            .reset_index()
        )

        return summaries, {}, results


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
        experiment_dir, overwrite=args.overwrite, overwrite_eval=args.overwrite_eval
    )

    # print results
    pprint(s)
    if args.plot:
        for name, fig in f.items():
            fig.canvas.manager.set_window_title(name)
        plt.show()
