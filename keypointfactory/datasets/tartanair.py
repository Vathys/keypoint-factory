import logging
from collections.abc import Iterable
import argparse

from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from omegaconf import OmegaConf

from ..settings import DATA_PATH
from ..geometry.wrappers import Camera, Pose
from ..utils.image import ImagePreprocessor, load_image
from ..utils.tools import fork_rng
from ..models.cache_loader import CacheLoader
from ..visualization.viz2d import plot_image_grid, plot_heatmaps
from .base_dataset import BaseDataset
from .utils import rotate_intrinsics, rotate_pose_inplane, scale_intrinsics

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
scene_lists_path = Path(__file__).parent / "tartanair_scene_lists"


def sample_n(data, num, seed=None):
    if len(data) > num:
        selected = np.random.RandomState(seed).choice(len(data), num, replace=False)
        return data[selected]
    else:
        return data


class TartanAir(BaseDataset):
    default_conf = {
        "data_dir": "tartan-air",
        "info_dir": "scene_info",
        "train_split": "tartanair_train_scenes.txt",
        "train_num_per_scene": None,
        "val_split": "tartanair_val_scenes.txt",
        "val_num_per_scene": None,
        "pair_types": "stereo_pairs",  # ["stereo_pairs", "random_pairs"]
        # data sampling
        "views": 2,
        # image options
        "read_depth": True,
        "read_image": True,
        "grayscale": False,
        "preprocessing": ImagePreprocessor.default_conf,
        "p_rotate": 0.0,
        "reseed": False,
        "seed": 0,
        "shuffle": True,
        "load_features": {
            "do": False,
            **CacheLoader.default_conf,
            "collate": False,
        },
    }

    def _init(self, conf):
        if not (DATA_PATH / conf.data_dir).exists():
            logger.info("Downloading TartanAir dataset")
            # self.download()

    def download(self):
        # Look at https://github.com/castacks/tartanair_tools
        # Might need to download on cluster and then preprocess the data with splits
        # Then link to that data for download instead.
        pass

    def get_dataset(self, split):
        assert self.conf.views in [1, 2, 3]
        if self.conf.views == 3:
            return _TripletDataset(self.conf, split)
        else:
            return _PairDataset(self.conf, split)


class _PairDataset(torch.utils.data.Dataset):
    def __init__(self, conf, split, load_sample=True):
        self.root = DATA_PATH / conf.data_dir
        assert self.root.exists(), self.root
        self.split = split
        self.conf = conf

        split_conf = conf[split + "_split"]
        if isinstance(split_conf, (str, Path)):
            scenes_path = scene_lists_path / split_conf
            scenes = scenes_path.read_text().rstrip("\n").split("\n")
        elif isinstance(split_conf, Iterable):
            scenes = list(split_conf)
        else:
            raise ValueError(f"Unknown split configuration: {split_conf}.")
        scenes = sorted(set(scenes))

        if conf.load_features.do:
            self.feature_loader = CacheLoader(conf.load_features)

        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        self.images = {}
        self.depths = {}
        self.poses = {}
        self.valid = {}
        self.g_camera_intrinsics = np.array(
            [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]]
        ).astype(np.float32, copy=False)

        self.info_dir = self.root / self.conf.info_dir
        self.scenes = []
        for scene in scenes:
            path = self.info_dir / (scene + ".npz")
            try:
                info = np.load(path, allow_pickle=True)
            except Exception:
                logger.warning("Cannot load scene info for scene %s at %s", scene, path)

            self.images[scene] = info["image_pairs"]
            self.depths[scene] = info["depth_pairs"]
            self.poses[scene] = info["poses"]
            self.scenes.append(scene)

        if load_sample:
            self.sample_new_items(conf.seed)
            assert len(self.items) > 0

    def sample_new_items(self, seed):
        logger.info("Sampling new %s data with seed %d", self.split, seed)
        self.items = []
        split = self.split
        num_per_scene = self.conf[split + "_num_per_scene"]

        if self.conf.views == 1:
            for scene in self.scenes:
                if scene not in self.images:
                    continue

                ids = np.arange(len(self.images[scene]))
                ids = np.random.RandomState(seed).choice(
                    ids, num_per_scene, replace=False
                )
                sides = np.random.RandomState(seed).choice(
                    [0, 1], num_per_scene, replace=True
                )
                ids = [(scene, i, side, i, side) for i, side in zip(ids, sides)]
                self.items.extend(ids)
        elif self.conf.views == 2:
            for scene in self.scenes:
                if scene not in self.images:
                    continue

                if self.conf.pair_types == "stereo_pairs":
                    ids = np.arange(len(self.images[scene]))
                    if num_per_scene and len(ids) > num_per_scene:
                        ids = np.random.RandomState(seed).choice(
                            ids, num_per_scene, replace=False
                        )
                    ids = [(scene, i, 0, i, 1) for i in ids]
                else:
                    assert self.conf.pair_types == "random_pairs"
                    ids = np.arange(len(self.images[scene]))

                    path = self.info_dir / (scene + ".npz")
                    info = np.load(path, allow_pickle=True)
                    mat = info["overlap_matrix"]

                    pairs = np.stack(np.where(mat == 1), -1)
                    if len(pairs) > num_per_scene:
                        pairs = sample_n(pairs, num_per_scene, seed)
                        side0 = np.random.RandomState(seed).choice(
                            [0, 1], num_per_scene, replace=True
                        )
                        side1 = np.random.RandomState(seed + 1).choice(
                            [0, 1], num_per_scene, replace=True
                        )
                    else:
                        pairs = np.stack(np.where(mat == 1), -1)
                        side0 = np.random.RandomState(seed).choice(
                            [0, 1], len(pairs), replace=True
                        )
                        side1 = np.random.RandomState(seed + 1).choice(
                            [0, 1], len(pairs), replace=True
                        )
                    ids = [
                        (scene, pair[0], si, pair[1], sj)
                        for pair, si, sj in zip(pairs, side0, side1)
                    ]
                self.items.extend(ids)
        if split == "train" and self.conf.shuffle:
            np.random.RandomState(seed).shuffle(self.items)

    def _read_view(self, scene, idx, side):
        path = self.root / self.images[scene][idx][side]

        K = self.g_camera_intrinsics
        T = self.poses[scene][idx][side].astype(np.float32, copy=False)

        if self.conf.read_image:
            img = load_image(path, self.conf.grayscale)
        else:
            size = PIL.Image.open(path).size[::-1]
            img = torch.zeros(
                3 - 2 * int(self.conf.grayscale), size[0], size[1]
            ).float()

        if self.conf.read_depth:
            depth_path = self.root / self.depths[scene][idx][side]
            depth = torch.Tensor(np.load(depth_path))[None]
            assert depth.shape[-2:] == img.shape[-2:]
        else:
            depth = None

        do_rotate = self.conf.p_rotate > 0 and self.split == "train"
        if do_rotate:
            p = self.conf.p_rotate
            k = 0
            if np.random.rand() < p:
                k = np.random.choice(2, 1, replace=False)[0] * 2 - 1
                img = np.rot90(img, k=-k, axes=(-2, -1))
                if self.conf.read_depth:
                    depth = np.rot90(depth, k=-k, axes=(-2, -1))
                K = rotate_intrinsics(K, img.shape, k + 2)
                T = rotate_pose_inplane(T, k + 2)

        name = f"{path.name}_{side}"

        data = self.preprocessor(img)
        if depth is not None:
            data["depth"] = self.preprocessor(depth, interpolation="nearest")["image"][
                0
            ]

        K = scale_intrinsics(K, data["scales"])

        data = {
            "name": name,
            "scene": scene,
            "T_w2cam": Pose.from_4x4mat(T),
            "depth": 80.0 / depth,
            "camera": Camera.from_calibration_matrix(K).float(),
            **data,
        }

        if self.conf.load_features.do:
            features = self.feature_loader({k: [v] for k, v in data.items()})
            if do_rotate and k != 0:
                # ang = np.deg2rad(k * 90.)
                kpts = features["keypoints"].copy()
                x, y = kpts[:, 0].copy(), kpts[:, 1].copy()
                w, h = data["image_size"]
                if k == 1:
                    kpts[:, 0] = w - y
                    kpts[:, 1] = x
                elif k == -1:
                    kpts[:, 0] = y
                    kpts[:, 1] = h - x

                else:
                    raise ValueError
                features["keypoints"] = kpts

            data = {"cache": features, **data}
        return data

    def __getitem__(self, idx):
        if self.conf.reseed:
            with fork_rng(self.conf.seed + idx, False):
                return self.getitem(idx)
        else:
            return self.getitem(idx)

    def getitem(self, idx):
        if self.conf.views == 2:
            if isinstance(idx, list):
                scene, idx0, side0, idx1, side1 = idx
            else:
                scene, idx0, side0, idx1, side1 = self.items[idx]
            data0 = self._read_view(scene, idx0, side0)
            data1 = self._read_view(scene, idx1, side1)
            data = {
                "view0": data0,
                "view1": data1,
            }
            data["T_0to1"] = data1["T_w2cam"] @ data0["T_w2cam"].inv()
            data["T_1to2"] = data0["T_w2cam"] @ data1["T_w2cam"].inv()
            data["name"] = f"{scene}/{data0['name']}-{data1['name']}"
        else:
            assert self.conf.views == 1
            scene, idx0, side0, idx1, side1 = self.items[idx]
            data = self._read_view(scene, idx0, side0)
            data = {"view0": data, "name": data["name"]}

        data["scene"] = scene
        data["idx"] = idx
        return data

    def __len__(self):
        return len(self.items)


class _TripletDataset(_PairDataset):
    def sample_new_items(self, seed):
        logger.info("Sampling new %s triplets with seed %d", self.split, seed)
        self.items = []
        split = self.split
        num_per_scene = self.conf[split + "_num_per_scene"]

        for scene in self.scenes:
            if scene not in self.images:
                continue

            assert self.conf.pair_types == "random_pairs"
            ids = ids0 = ids1 = ids2 = np.arange(len(self.images[scene]))

            ids0 = np.random.RandomState(seed).choice(
                ids, (num_per_scene,), replace=False
            )
            side0 = np.random.RandomState(seed).choice(
                [0, 1], num_per_scene, replace=True
            )
            ids1 = np.random.RandomState(seed + 1).choice(
                ids, num_per_scene, replace=False
            )
            side1 = np.random.RandomState(seed + 1).choice(
                [0, 1], num_per_scene, replace=True
            )
            ids2 = np.random.RandomState(seed + 2).choice(
                ids, num_per_scene, replace=False
            )
            side2 = np.random.RandomState(seed + 2).choice(
                [0, 1], num_per_scene, replace=True
            )
            ids = [
                (scene, i, si, j, sj, k, sk)
                for i, si, j, sj, k, sk in zip(ids0, side0, ids1, side1, ids2, side2)
            ]
            self.items.extend(ids)

        if split == "train" and self.conf.shuffle:
            np.random.RandomState(seed).shuffle(self.items)

    def __getitem__(self, idx):
        scene, idx0, side0, idx1, side1, idx2, side2 = self.items[idx]

        data0 = self._read_view(scene, idx0, side0)
        data1 = self._read_view(scene, idx1, side1)
        data2 = self._read_view(scene, idx2, side2)
        data = {
            "view0": data0,
            "view1": data1,
            "view2": data2,
        }
        data["T_0to1"] = data1["T_w2cam"] @ data0["T_w2cam"].inv()
        data["T_0to2"] = data2["T_w2cam"] @ data0["T_w2cam"].inv()
        data["T_1to2"] = data2["T_w2cam"] @ data1["T_w2cam"].inv()
        data["T_1to0"] = data0["T_w2cam"] @ data1["T_w2cam"].inv()
        data["T_2to0"] = data0["T_w2cam"] @ data2["T_w2cam"].inv()
        data["T_2to1"] = data1["T_w2cam"] @ data2["T_w2cam"].inv()

        data["scene"] = scene
        data["name"] = f"{scene}/{data0['name']}-{data1['name']}-{data2['name']}"
        return data


def visualize(args):
    conf = {
        "train_num_per_scene": 5,
        "batch_size": 1,
        "num_workers": 0,
        "prefetch_factor": None,
        "val_num_per_scene": None,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = TartanAir(conf)
    loader = dataset.get_data_loader(args.split)
    logger.info(f"The dataset has elements. {len(loader)}")

    with fork_rng(seed=dataset.conf.seed):
        images, depths = [], []
        for _, data in zip(range(args.num_items), loader):
            images.append(
                [
                    data[f"view{i}"]["image"][0].permute(1, 2, 0)
                    for i in range(dataset.conf.views)
                ]
            )
            depths.append(
                [data[f"view{i}"]["depth"][0] for i in range(dataset.conf.views)]
            )

    axes = plot_image_grid(images, dpi=args.dpi)
    for i in range(len(images)):
        plot_heatmaps(depths[i], axes=axes[i])
    plt.show()


if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_items", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    visualize(args)
