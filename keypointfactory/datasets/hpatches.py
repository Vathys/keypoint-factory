"""
Simply load images from a folder or nested folders (does not have any split).
"""

import argparse
import logging
import tarfile
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from ..settings import DATA_PATH
from ..utils.image import ImagePreprocessor, load_image
from ..utils.tools import fork_rng
from ..visualization.viz2d import plot_image_grid
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


def read_homography(path):
    with open(path) as f:
        result = []
        for line in f.readlines():
            while "  " in line:  # Remove double spaces
                line = line.replace("  ", " ")
            line = line.replace(" \n", "").replace("\n", "")
            # Split and discard empty strings
            elements = list(filter(lambda s: s, line.split(" ")))
            if elements:
                result.append(elements)
        return np.array(result).astype(float)


class HPatches(BaseDataset, torch.utils.data.Dataset):
    default_conf = {
        "preprocessing": ImagePreprocessor.default_conf,
        "data_dir": "hpatches-sequences-release",
        "subset": None,
        "ignore_large_images": True,
        "grayscale": False,
        "triplets": False,
    }

    # Large images that were ignored in previous papers
    ignored_scenes = (
        "i_contruction",
        "i_crownnight",
        "i_dc",
        "i_pencils",
        "i_whitebuilding",
        "v_artisans",
        "v_astronautis",
        "v_talent",
    )
    url = "http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz"

    def _init(self, conf):
        # assert conf.batch_size == 1
        self.preprocessor = ImagePreprocessor(conf.preprocessing)

        self.root = DATA_PATH / conf.data_dir
        if not self.root.exists():
            logger.info("Downloading the HPatches dataset.")
            self.download()
        self.sequences = sorted([x.name for x in self.root.iterdir()])
        if not self.sequences:
            raise ValueError("No image found!")
        self.items = []  # (seq, q_idx, is_illu)
        for seq in self.sequences:
            if conf.ignore_large_images and seq in self.ignored_scenes:
                continue
            if conf.subset is not None and conf.subset != seq[0]:
                continue
            if self.conf.triplets:
                self.items.extend([(seq, i, j, seq[0] == "i") for i, j in combinations(list(range(2, 7)), 2)])
            else:
                self.items.extend([(seq, i, seq[0] == "i") for i in range(2, 7)])
        logger.info(f"Loaded {len(self.items)} items.")

    def download(self):
        data_dir = self.root.parent
        data_dir.mkdir(exist_ok=True, parents=True)
        tar_path = data_dir / self.url.rsplit("/", 1)[-1]
        torch.hub.download_url_to_file(self.url, tar_path)
        with tarfile.open(tar_path) as tar:
            tar.extractall(data_dir)
        tar_path.unlink()

    def get_dataset(self, split):
        # Stops testin go for now
        # Returns the same dataset for training, validation and testing
        # assert split in ["val", "test"]
        return self

    def _read_image(self, seq: str, idx: int) -> dict:
        img = load_image(self.root / seq / f"{idx}.ppm", self.conf.grayscale)
        return self.preprocessor(img)

    def __getitem__(self, idx):
        if self.conf.triplets:
            seq, q1_idx, q2_idx, is_illu = self.items[idx]
            data0 = self._read_image(seq, 1)
            data1 = self._read_image(seq, q1_idx)
            data2 = self._read_image(seq, q2_idx)
            H_1 = read_homography(self.root / seq / f"H_1_{q1_idx}")
            H_2 = read_homography(self.root / seq / f"H_1_{q2_idx}")
            H_0to1 = data1["transform"] @ H_1 @ np.linalg.inv(data0["transform"])
            H_0to2 = data2["transform"] @ H_2 @ np.linalg.inv(data0["transform"])
            H_1to2 = data2["transform"] @ np.linalg.inv(H_1) @ H_2 @ np.linalg.inv(data1["transform"])
            return {
                "view0": data0,
                "view1": data1,
                "view2": data2,
                "scene": seq,
                "idx": idx,
                "is_illu": is_illu,
                "name": f"{seq}/{idx}.ppm",
                "H_0to1": H_0to1.astype(np.float32),
                "H_0to2": H_0to2.astype(np.float32),
                "H_1to2": H_1to2.astype(np.float32),
            }
        else:
            seq, q_idx, is_illu = self.items[idx]
            data0 = self._read_image(seq, 1)
            data1 = self._read_image(seq, q_idx)
            H = read_homography(self.root / seq / f"H_1_{q_idx}")
            H = data1["transform"] @ H @ np.linalg.inv(data0["transform"])
            return {
                "H_0to1": H.astype(np.float32),
                "scene": seq,
                "idx": idx,
                "is_illu": is_illu,
                "name": f"{seq}/{idx}.ppm",
                "view0": data0,
                "view1": data1,
            }

    def __len__(self):
        return len(self.items)


def visualize(args):
    conf = {
        "batch_size": 1,
        "num_workers": 8,
        "prefetch_factor": 1,
    }
    conf = OmegaConf.merge(conf, OmegaConf.from_cli(args.dotlist))
    dataset = HPatches(conf)
    loader = dataset.get_data_loader("test")
    logger.info("The dataset has %d elements.", len(loader))

    with fork_rng(seed=dataset.conf.seed):
        images = []
        for _, data in zip(range(args.num_items), loader):
            images.append(
                [data[f"view{i}"]["image"][0].permute(1, 2, 0) for i in range(2)]
            )
    plot_image_grid(images, dpi=args.dpi)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from .. import logger  # overwrite the logger

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_items", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()
    visualize(args)
