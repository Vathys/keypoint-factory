import pydegensac
import torch

from ... import logger
from ..base_estimator import BaseEstimator


class DegensacHomographyEstimator(BaseEstimator):
    default_conf = {
        "ransac_th": 2.0,
        "options": {
            "confidence": 0.9999,
            "max_iters": 10_000,
            "candidate_threshold": 10,
        },
    }

    required_data_keys = ["m_kpts0", "m_kpts1"]

    def _init(self, conf):
        pass

    def _forward(self, data):
        pts0, pts1 = data["m_kpts0"], data["m_kpts1"]

        if (
            pts0.shape[1] < self.conf.options["candidate_threshold"]
            or pts1.shape[1] < self.conf.options["candidate_threshold"]
        ):
            return {
                "success": False,
                "M_0to1": torch.eye(3, device=pts0.device, dtype=pts0.dtype),
                "inliers": torch.tensor([]).to(pts0.device),
            }

        M, mask = pydegensac.findHomography(
            pts0.squeeze(0).cpu().numpy(),
            pts1.squeeze(0).cpu().numpy(),
            px_th=self.conf.ransac_th,
            conf=self.conf.options["confidence"],
            max_iters=self.conf.options["max_iters"],
        )

        if mask is None:
            logger.warning("Degensac failed to find a solution")
            return {
                "success": False,
                "M_0to1": torch.eye(3, device=pts0.device, dtype=pts0.dtype),
                "inliers": torch.tensor([]).to(pts0.device),
            }

        mask = torch.from_numpy(mask).to(pts0.device)
        M = torch.from_numpy(M).float().to(pts0.device)

        return {
            "success": True,
            "M_0to1": M,
            "inliers": mask.to(pts0),
        }
