import torch
import pydegensac
from ...geometry.wrappers import Pose
from ...geometry.epipolar import F_to_E
from ..base_estimator import BaseEstimator
from kornia.geometry.epipolar import motion_from_essential_choose_solution
from ... import logger


class DegensacPoseEstimator(BaseEstimator):
    default_conf = {
        "ransac_th": 2.0,
        "options": {
            "confidence": 0.9999,
            "max_iters": 10_000,
            "candidate_threshold": 10,
        },
    }

    # require depth or homography? Calcualte matches ourselves?
    required_data_keys = ["m_kpts0", "m_kpts1", "camera0", "camera1"]

    def _init(self, conf):
        pass

    def _forward(self, data):
        pts0, pts1 = data["m_kpts0"], data["m_kpts1"]
        # Make sure it's a batched tensor of size 1
        assert len(pts0.shape) == 3 and pts0.shape[0] == 1
        camera0 = data["camera0"]
        camera1 = data["camera1"]

        K0 = camera0.calibration_matrix().unsqueeze(0)
        K1 = camera1.calibration_matrix().unsqueeze(0)

        if pts0.shape[1] < self.conf.options["candidate_threshold"]:
            return {
                "success": False,
                "M_0to1": Pose.from_4x4mat(torch.eye(4)).to(pts0),
                "inliers": torch.tensor([]).to(pts0),
            }

        F, mask = pydegensac.findFundamentalMatrix(
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
                "M_0to1": Pose.from_4x4mat(torch.eye(4)).to(pts0),
                "inliers": torch.tensor([]).to(pts0),
            }

        mask = torch.from_numpy(mask)
        F = torch.from_numpy(F).unsqueeze(0).to(pts0)

        E = F_to_E(camera0, camera1, F)

        R, t, _ = motion_from_essential_choose_solution(
            E,
            K0,
            K1,
            pts0,
            pts1,
            mask.unsqueeze(0),
        )

        return {
            "success": True,
            "M_0to1": Pose.from_Rt(R.squeeze(), t.squeeze()).to(pts0),
            "inliers": mask.to(pts0),
        }
