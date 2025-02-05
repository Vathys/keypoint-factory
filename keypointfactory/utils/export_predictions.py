"""
Export the predictions of a model for a given dataloader (e.g. ImageFolder).
Use a standalone script with `python3 -m dsfm.scipts.export_predictions dir`
or call from another script.
"""

from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from .tensor import batch_to_device


@torch.no_grad()
def export_predictions(
    loader,
    model,
    output_file,
    as_half=False,
    keys="*",
    callback_fn=None,
    optional_keys=[],
):
    assert keys == "*" or isinstance(keys, (tuple, list))
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    with h5py.File(str(output_file), "w") as hfile:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        for data_ in tqdm(loader):
            data = batch_to_device(data_, device, non_blocking=True)
            pred = model(data)
            if callback_fn is not None:
                pred = {**callback_fn(pred, data), **pred}
            if keys != "*":
                if len(set(keys) - set(pred.keys())) > 0:
                    raise ValueError(f"Missing key {set(keys) - set(pred.keys())}")
                pred = {k: v for k, v in pred.items() if k in keys + optional_keys}
            assert len(pred) > 0

            # renormalization
            for k in pred.keys():
                if k.startswith("keypoints"):
                    idx = k.replace("keypoints", "")
                    scales = 1.0 / (
                        data["scales"]
                        if len(idx) == 0
                        else data[f"view{idx}"]["scales"]
                    )
                    pred[k] = pred[k] * scales[None]
                if k.startswith("lines"):
                    idx = k.replace("lines", "")
                    scales = 1.0 / (
                        data["scales"]
                        if len(idx) == 0
                        else data[f"view{idx}"]["scales"]
                    )
                    pred[k] = pred[k] * scales[None]
                if k.startswith("orig_lines"):
                    idx = k.replace("orig_lines", "")
                    scales = 1.0 / (
                        data["scales"]
                        if len(idx) == 0
                        else data[f"view{idx}"]["scales"]
                    )
                    pred[k] = pred[k] * scales[None]

            pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

            if as_half:
                for k in pred:
                    dt = pred[k].dtype
                    if (dt == np.float32) and (dt != np.float16):
                        pred[k] = pred[k].astype(np.float16)
            try:
                name = data["name"][0]
                grp = hfile.create_group(name)
                for k, v in pred.items():
                    grp.create_dataset(k, data=v)
            except RuntimeError:
                continue

            del pred
    return output_file


@torch.no_grad()
def export_transformed_predictions(
    loader, model, transform, output_file, keys="*", optional_keys=[]
):
    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    hfile = h5py.File(str(output_file), "w")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    for data_ in tqdm(loader, position=0):
        data_pred = {}

        for tdata_ in transform(data_):
            tdata_ = batch_to_device(tdata_, device, non_blocking=True)
            pred = model(tdata_)

            if keys != "*":
                if len(set(keys) - set(pred.keys())) > 0:
                    raise ValueError(f"Missing key {set(keys) - set(pred.keys())}")
                pred = {k: v for k, v in pred.items() if k in keys + optional_keys}
            assert len(pred) > 0

            pred["transform"] = tdata_["transform"]

            for k in pred.keys():
                if k.startswith("keypoints"):
                    idx = k.replace("keypoints", "")
                    scales = 1.0 / (
                        tdata_["scales"]
                        if len(idx) == 0
                        else tdata_[f"view{idx}"]["scales"]
                    )
                    pred[k] = pred[k] * scales[None]
                if k.startswith("lines"):
                    idx = k.replace("lines", "")
                    scales = 1.0 / (
                        tdata_["scales"]
                        if len(idx) == 0
                        else tdata_[f"view{idx}"]["scales"]
                    )
                    pred[k] = pred[k] * scales[None]
                if k.startswith("orig_lines"):
                    idx = k.replace("orig_lines", "")
                    scales = 1.0 / (
                        tdata_["scales"]
                        if len(idx) == 0
                        else tdata_[f"view{idx}"]["scales"]
                    )
                    pred[k] = pred[k] * scales[None]

            for k, v in pred.items():
                if k not in data_pred:
                    data_pred[k] = []
                data_pred[k].append(v[0].cpu().numpy())

        try:
            name = data_["name"][0]
            grp = hfile.create_group(name)
            for k, v in data_pred.items():
                if k in ["keypoints0", "keypoint_scores0", "heatmap0"]:
                    grp.create_dataset(k, data=v[0])
                else:
                    subgrp = grp.create_group(k)
                    for i, vv in enumerate(v):
                        subgrp.create_dataset(str(i), data=vv)
        except RuntimeError:
            continue

        del data_pred

    hfile.close()
    return output_file
