import logging
import zipfile
from pathlib import Path
from pprint import pprint

import torch
from omegaconf import OmegaConf

from .eval_pipeline import EvalPipeline
from ..settings import DATA_PATH, EVAL_PATH

from .io import get_eval_parser, parse_eval_args


logger = logging.getLogger(__name__)


class Scannet1500Pipeline(EvalPipeline):
    default_conf = {
        "data": {
            "name": "image_pairs",
            "pairs": "scannet1500/pairs_calibrated.txt",
            "root": "scannet1500/",
            "extra_data": "relative_pose",
            "preprocessing": {
                "side": "long",
            },
        }
    }

    export_keys = [
        "keypoints0",
        "keypoints1",
        "keypoint_scores0",
        "keypoint_scores1",
    ]

    def _init(self, conf):
        if not (DATA_PATH / "scannet1500").exists():
            logger.info("Downloading the ScanNet-1500 dataset.")
            url = "https://cvg-data.inf.ethz.ch/scannet/scannet1500.zip"
            zip_path = DATA_PATH / url.rsplit("/", 1)[-1]
            zip_path.parent.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(url, zip_path)
            with zipfile.ZipFile(zip_path) as fid:
                fid.extractall(DATA_PATH)
            zip_path.unlink()

    @classmethod



if __name__ == "__main__":
    from .. import logger

    dataset_name = Path(__file__).stem
    parser = get_eval_parser()
    args = parser.parse_intermixed_args()

    default_conf = OmegaConf.create(Scannet1500Pipeline.default_conf)

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

    pipeline = Scannet1500Pipeline(conf)
    # s, f, r = pipeline.run(
    #     experiment_dir,
    #     overwrite=args.overwrite,
    #     overwrite_eval=args.overwrite_eval,
    # )

    # pprint(s)
