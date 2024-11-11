"""
A two-view sparse feature matching pipeline on triplets.

If a triplet is found, runs the extractor on three images and
then runs matcher/filter/solver for all three pairs.

Losses and metrics get accumulated accordingly.

If no triplet is found, this falls back to two_view_pipeline.py
"""

import torch

from ..utils.misc import get_twoview, stack_twoviews
from .two_view_pipeline import TwoViewPipeline

from .. import logger

def has_triplet(data):
    # we already check for image0 and image1 in required_keys
    return "view2" in data.keys()


class TripletPipeline(TwoViewPipeline):
    default_conf = {
        "batch_triplets": False,
        "enumerate_pairs": True,
        **TwoViewPipeline.default_conf,
    }

    def _forward(self, data):
        if not has_triplet(data):
            raise RuntimeError("No triplets found.")
            # return super()._forward(data)
        # the two-view outputs are stored in
        # pred['0to1'],pred['0to2'], pred['1to2']

        assert not self.conf.run_gt_in_forward
        assert not self.conf.pass_all_views

        pred0 = self.extract_view(data, "0")
        pred1 = self.extract_view(data, "1")
        pred2 = self.extract_view(data, "2")

        all_pred = {
            **{k + "0": v for k, v in pred0.items()},
            **{k + "1": v for k, v in pred1.items()},
            **{k + "2": v for k, v in pred2.items()},
        }

        if self.conf.enumerate_pairs:
            pred = {}
            for idx in ["0to1", "0to2", "1to2"]:
                pred[idx] = get_twoview(all_pred, idx)
            return pred
        else:
            return all_pred

    def _pre_loss_callback(self, seed, epoch):
        super()._pre_loss_callback(seed, epoch)

    def _post_loss_callback(self, seed, epoch):
        super()._post_loss_callback(seed, epoch)

    def _detach_grad_filter(self, key):
        return super()._detach_grad_filter(key)

    def loss(self, pred, data):
        if not has_triplet(data):
            raise RuntimeError("No triplets found.")
            # return super().loss(pred, data)1

        losses = {}
        metrics = {}
        if self.conf.enumerate_pairs:
            if self.conf.batch_triplets:
                # TODO: stacking doesn't work for Pose and Camera wrappers.
                m_data = stack_twoviews(data)
                m_pred = stack_twoviews(pred)
                losses, metrics = super().loss(m_pred, m_data)
            else:
                for idx in ["0to1", "0to2", "1to2"]:
                    data_i = get_twoview(data, idx)
                    pred_i = pred[idx]
                    losses_i, metrics_i = super().loss(pred_i, data_i)
                    for k, v in losses_i.items():
                        if k in losses.keys():
                            losses[k].append(v)
                        else:
                            losses[k] = [v]
                    for k, v in metrics_i.items():
                        if k in metrics.keys():
                            metrics[k].append(v)
                        else:
                            metrics[k] = [v]
        else:
            loss, metric = super().loss(pred, data)
            for k, v in loss.items():
                losses[k] = [v]
            for k, v in metric.items():
                metrics[k] = [v]

        for k in losses.keys():
            losses[k] = torch.cat(losses[k], 0)
        for k in metrics.keys():
            metrics[k] = torch.cat(metrics[k], 0)

        return losses, metrics
