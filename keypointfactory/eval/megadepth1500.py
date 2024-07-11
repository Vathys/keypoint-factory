import logging
import zipfile
from typing import Iterable
from collections import defaultdict
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ..datasets import get_dataset
from ..models.cache_loader import CacheLoader
from ..settings import DATA_PATH, EVAL_PATH
from ..utils.export_predictions import export_predictions
from ..utils.tensor import map_tensor
from .eval_pipeline import EvalPipeline
from .io import get_eval_parser, load_model, parse_eval_args
from .utils import eval_pair_depth, eval_relative_pose_robust


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

    def get_predictions(self, experiment_dir, model=None, overwrite=False):
        """Export a prediction file for each eval datapoint"""
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
        """Run the eval on cached predictions"""
        assert pred_file.exists()
        results = defaultdict(list)
        conf = self.conf.eval

        if isinstance(conf.top_k_thresholds, int) or conf.top_k_thresholds is None:
            conf.top_k_thresholds = [conf.top_k_thresholds]
        if isinstance(conf.top_k_by, str) or conf.top_k_by is None:
            conf.top_k_by = [conf.top_k_by]

        results = defaultdict(list)
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
            pred = map_tensor(pred, lambda x: torch.squeeze(x, dim=0))
            scene_name = data["name"][0].split("-")[0]
            for top_k in conf.top_k_thresholds:
                for top_by in conf.top_k_by:
                    pair_metrics = eval_pair_depth(
                        data,
                        pred,
                        eval_to_0=False,
                        top_k=int(top_k),
                        top_by=top_by,
                        thresh=conf.correctness_threshold,
                        padding=conf.padding,
                    )
                    pair_metrics["top_k"] = top_k
                    pair_metrics["top_by"] = top_by
                    pair_metrics["scene"] = scene_name
                    if self.conf.eval.estimator:
                        for th in test_thresholds:
                            pose_metrics = eval_relative_pose_robust(
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
                        self.conf.model.max_num_keypoints
                        if self.conf.model.max_num_keypoints is not None
                        else float("inf")
                    )
                )
            )
        )  # Multiple top_k by 2 because we take
        # sum of correct points from two images

        def calc_auc(df):
            hist, _ = np.histogram(
                np.nan_to_num(
                    df.to_numpy(np.float32), nan=0, posinf=179.95, neginf=179.95
                ),
                bins=10,
            )
            hist = hist / df.shape[0]
            auc = hist.cumsum().mean()
            return auc

        groupby_columns = ["top_k", "top_by"]
        agg_funcs = {
            "num_keypoints": ("num_keypoints", "sum"),
            "num_covisible": ("num_covisible", "sum"),
            "num_covisible_correct": ("num_covisible_correct", "sum"),
            "localization_score": ("localization_score", "sum"),
            "repeatability": ("repeatability", "mean"),
        }

        if self.conf.eval.estimator:
            agg_funcs["rel_pose_auc"] = ("rel_pose_error", calc_auc)
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

        summaries = (
            results.groupby(["scene", "top_k", "top_by", "ransac_th"])
            .agg(
                num_keypoints=pd.NamedAgg(column="num_keypoints", aggfunc="sum"),
                num_covisible=pd.NamedAgg(column="num_covisible", aggfunc="sum"),
                num_covisible_correct=pd.NamedAgg(
                    column="num_covisible_correct", aggfunc="sum"
                ),
                localization_score=pd.NamedAgg(
                    column="localization_score", aggfunc="sum"
                ),
                repeatability=pd.NamedAgg(column="repeatability", aggfunc="mean"),
                rel_pose_auc=pd.NamedAgg(column="rel_pose_error", aggfunc=calc_auc),
            )
            .reset_index()
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
