from collections import defaultdict
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import cv2 as cv
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from ..datasets import get_dataset
from ..settings import EVAL_PATH
from ..models.cache_loader import CacheLoader
from .utils import eval_pair_homography
from ..utils.export_predictions import export_transformed_predictions
from ..utils.tensor import map_tensor
from ..geometry.homography import warp_points_torch
from .eval_pipeline import EvalPipeline
from .io import get_eval_parser, load_model, parse_eval_args


def perform_rotation(image, angle):
    height, width = image.shape[:2]
    rot_mat = cv.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    rot_image = cv.warpAffine(
        image,
        rot_mat,
        (width, height),
        flags=cv.INTER_LANCZOS4,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0,
    )
    return rot_image


def perform_crop(image, scale, resize=False):
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale), int(width * scale)
    crop_image = cv.getRectSubPix(
        image, (new_width, new_height), (width // 2, height // 2)
    )
    if resize:
        crop_image = cv.resize(
            crop_image, (width, height), interpolation=cv.INTER_LANCZOS4
        )

    return crop_image


def get_hpatches_scenes(data_loader, cache_loader):
    name = None
    group = []

    for i, data in enumerate(data_loader):
        if name is None:
            name = str(Path(data["name"][0]).parent)
            group.append({"data": data, "pred": cache_loader(data)})
            continue

        if str(Path(data["name"][0]).parent) == name:
            group.append({"data": data, "pred": cache_loader(data)})
        else:
            yield name, group
            name = str(Path(data["name"][0]).parent)
            group = [{"data": data, "pred": cache_loader(data)}]

    if len(group) > 0:
        yield name, group


def plot_summary(summary, transform, transform_points):
    fig, axes = plt.subplots(1, 1, figsize=(16, 12), sharey=True)

    reduced = (
        summary.groupby([transform, "scene_type"]).mean(numeric_only=True).reset_index()
    )

    illum = reduced[reduced["scene_type"] == "i"]
    view = reduced[reduced["scene_type"] == "v"]

    sns.lineplot(
        data=illum,
        x=transform,
        y="scene_repeatability",
        markers=True,
        dashes=False,
        ax=axes,
        label="illumination",
    )

    sns.lineplot(
        data=view,
        x=transform,
        y="scene_repeatability",
        markers=True,
        dashes=False,
        ax=axes,
        label="viewpoint",
    )

    sns.lineplot(
        data=reduced,
        x=transform,
        y="scene_repeatability",
        markers=True,
        dashes=False,
        ax=axes,
        label="all",
    )

    axes.legend()

    axes.set_title("Scene Repeatability")
    axes.set_xlabel(transform.capitalize())
    axes.set_ylabel("Repeatability")

    axes.set_xticks(transform_points, labels=transform_points)

    fig.tight_layout()

    return fig


class HPatchesTransformationsPipeline(EvalPipeline):
    default_conf = {
        "data": {
            "batch_size": 1,
            "name": "hpatches",
            "num_workers": 16,
            "preprocessing": {
                "resize": 480,
                "side": "short",
            },
        },
        "model": {
            "ground_truth": {
                "name": None,
            }
        },
        "eval": {
            "correctness_threshold": 3.0,
            "padding": 4.0,
            "transformation": "rotation",  # or scale
            "test_angles": [0, 45, 90, 135, 180, 225, 270, 315],
            "test_scales": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            "top_k_threshold": 2048,
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
        pass

    def perform_rotation_test(self, data):
        for angle in self.conf.eval.test_angles:
            data_ = map_tensor(data, lambda x: x.detach().clone())
            batch, _, height, width = data_["view1"]["image"].shape

            batch_list = []
            data_["transform"] = torch.zeros(batch, 3)
            for i in range(batch):
                timage = perform_rotation(
                    data_["view1"]["image"][i].permute(1, 2, 0).numpy(), angle
                )
                batch_list.append(torch.from_numpy(timage).permute(2, 0, 1))
                data_["transform"][i] = torch.tensor([[angle, 1, False]])

            if len(batch_list) > 1:
                data_["view1"]["image"] = torch.stack(batch_list, dim=0)
            else:
                data_["view1"]["image"] = batch_list[0].unsqueeze(0)

            yield data_

    def perform_scale_test(self, data, resize=True):
        for scale in self.conf.eval.test_scales:
            data_ = map_tensor(data, lambda x: x.detach().clone())
            batch, _, height, width = data_["view1"]["image"].shape

            batch_list = []
            data_["transform"] = torch.zeros(batch, 3)
            for i in range(batch):
                timage = perform_crop(
                    data_["view1"]["image"][i].permute(1, 2, 0).numpy(),
                    scale,
                    resize=resize,
                )
                batch_list.append(torch.from_numpy(timage).permute(2, 0, 1))
                data_["transform"][i] = torch.tensor([[0, scale, resize]])

            if len(batch_list) > 1:
                data_["view1"]["image"] = torch.stack(batch_list, dim=0)
            else:
                data_["view1"]["image"] = batch_list[0].unsqueeze(0)

            yield data_

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
                export_transformed_predictions(
                    self.get_dataloader(self.conf.data),
                    model,
                    (
                        self.perform_rotation_test
                        if self.conf.eval.transformation == "rotation"
                        else self.perform_scale_test
                    ),
                    pred_file,
                )

        return pred_file

    def run_eval(self, loader, pred_file):
        assert pred_file.exists()
        results = defaultdict(list)

        conf = self.conf.eval

        if conf.top_k_threshold is None:
            conf.top_k_threshold = 2048

        df_list = []
        cache_loader = CacheLoader({"path": str(pred_file), "collate": None}).eval()
        for scene_name, scene in tqdm(
            get_hpatches_scenes(loader, cache_loader), total=108
        ):
            for data_ in scene:
                data_ = map_tensor(data_, lambda x: torch.squeeze(x, dim=0))

                kpts0 = data_["pred"]["keypoints0"]
                kpts1 = data_["pred"]["keypoints1"]
                kpts_score0 = data_["pred"]["keypoint_scores0"]
                kpts_score1 = data_["pred"]["keypoint_scores1"]

                new_pred = data_["pred"]
                idxs0 = torch.argsort(kpts_score0, descending=True)[
                    : conf.top_k_threshold
                ]
                kpts1_ = {}
                kpts_score1_ = {}
                for i in kpts_score1.keys():
                    idxs = torch.argsort(kpts_score1[i], descending=True)[
                        : conf.top_k_threshold
                    ]
                    kpts1_[i] = kpts1[i][idxs]
                    kpts_score1_[i] = kpts_score1[i][idxs]

                new_pred["keypoints0"] = kpts0[idxs0]
                new_pred["keypoints1"] = kpts1_
                new_pred["keypoint_scores0"] = kpts_score0[idxs0]
                new_pred["keypoint_scores1"] = kpts_score1_

                res_dict = eval_pair_homography(
                    data_["data"],
                    new_pred,
                    eval_to_0=True,
                    thresh=conf.correctness_threshold,
                    padding=conf.padding,
                )
                res_dict["scene_name"] = scene_name
                df_list.append(res_dict)

        results = pd.concat(df_list)

        summary = (
            results.groupby(["rotation", "scale", "scene_name"])
            .sum(numeric_only=True)
            .reset_index()
        )

        summary["scene_repeatability"] = (
            summary["num_covisible_correct"] / summary["num_covisible"]
        )
        summary["scene_localization"] = (
            summary["localization_score"] / summary["num_covisible_correct"]
        )

        summary["scene_type"] = summary["scene_name"].apply(lambda x: x.split("_")[0])

        figures = {}
        figures["summary_graph"] = plot_summary(
            summary,
            conf.transformation,
            conf.test_angles if conf.transformation == "rotation" else conf.test_scales,
        )

        return summary, figures, results


if __name__ == "__main__":
    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(HPatchesTransformationsPipeline.default_conf)

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

    pipeline = HPatchesTransformationsPipeline(conf)
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
