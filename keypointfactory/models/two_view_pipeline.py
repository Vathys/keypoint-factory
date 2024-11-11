"""
A two-view sparse feature matching pipeline.

This model contains sub-models for each step:
    feature extraction, feature matching, outlier filtering, pose estimation.
Each step is optional, and the features or matches can be provided as input.
Default: SuperPoint with nearest neighbor matching.

Convention for the matches: m0[i] is the index of the keypoint in image 1
that corresponds to the keypoint i in image 0. m0[i] = -1 if i is unmatched.
"""

from omegaconf import OmegaConf

from . import get_model
from .base_model import BaseModel

to_ctr = OmegaConf.to_container  # convert DictConfig to dict


class TwoViewPipeline(BaseModel):
    default_conf = {
        "extractor": {
            "name": None,
            "trainable": False,
        },
        "filter": {"name": None},
        "solver": {"name": None},
        "ground_truth": {"name": None},
        "allow_no_extract": False,
        "run_gt_in_forward": False,
        "pass_all_views": False,
    }
    required_data_keys = ["view0", "view1"]
    strict_conf = False  # need to pass new confs to children models
    components = [
        "extractor",
        "filter",
        "solver",
        "ground_truth",
    ]

    def _init(self, conf):
        if conf.extractor.name:
            self.extractor = get_model(conf.extractor.name)(to_ctr(conf.extractor))

        if conf.filter.name:
            self.filter = get_model(conf.filter.name)(to_ctr(conf.filter))

        if conf.solver.name:
            self.solver = get_model(conf.solver.name)(to_ctr(conf.solver))

        if conf.ground_truth.name:
            self.ground_truth = get_model(conf.ground_truth.name)(
                to_ctr(conf.ground_truth)
            )

    def extract_view(self, data, i):
        data_i = data[f"view{i}"]
        pred_i = data_i.get("cache", {})
        skip_extract = len(pred_i) > 0 and self.conf.allow_no_extract
        if self.conf.extractor.name and not skip_extract:
            pred_i = {**pred_i, **self.extractor(data_i)}
        elif self.conf.extractor.name and not self.conf.allow_no_extract:
            pred_i = {**pred_i, **self.extractor({**data_i, **pred_i})}
        return pred_i

    def _pre_loss_callback(self, seed, epoch):
        super()._pre_loss_callback(seed, epoch)

        if self.conf.extractor.name:
            self.extractor._pre_loss_callback(seed, epoch)
        if self.conf.filter.name:
            self.filter._pre_loss_callback(seed, epoch)
        if self.conf.solver.name:
            self.solver._pre_loss_callback(seed, epoch)

    def _post_loss_callback(self, seed, epoch):
        super()._post_loss_callback(seed, epoch)

        if self.conf.extractor.name:
            self.extractor._post_loss_callback(seed, epoch)
        if self.conf.filter.name:
            self.filter._post_loss_callback(seed, epoch)
        if self.conf.solver.name:
            self.solver._post_loss_callback(seed, epoch)

    def _detach_grad_filter(self, key):
        detach = True
        if self.conf.extractor.name:
            detach = detach and self.extractor._detach_grad_filter(key)
        if self.conf.filter.name:
            detach = detach and self.filter._detach_grad_filter(key)
        if self.conf.solver.name:
            detach = detach and self.solver._detach_grad_filter(key)
        return detach

    def _forward(self, data):
        if self.conf.pass_all_views:
            pred = self.extractor(data)
        else:
            pred0 = self.extract_view(data, "0")
            pred1 = self.extract_view(data, "1")
            pred = {
                **{k + "0": v for k, v in pred0.items()},
                **{k + "1": v for k, v in pred1.items()},
            }

        # Remove matcher component to avoid later confusion?
        if self.conf.filter.name:
            pred = {**pred, **self.filter({**data, **pred})}
        if self.conf.solver.name:
            pred = {**pred, **self.solver({**data, **pred})}

        return pred

    def loss(self, pred, data):
        losses = {}
        metrics = {}

        # get labels
        for k in self.components:
            apply = True
            if "apply_loss" in self.conf[k].keys():
                apply = self.conf[k].apply_loss
            if self.conf[k].name and apply:
                try:
                    losses_, metrics_ = getattr(self, k).loss(pred, data)
                except NotImplementedError:
                    continue
                losses = {**losses, **losses_}
                metrics = {**metrics, **metrics_}

        return losses_, metrics_
