import kornia
import torch

from .utils import get_image_coords
from .wrappers import Camera


def sample_fmap(pts, fmap):
    h, w = fmap.shape[-2:]
    grid_sample = torch.nn.functional.grid_sample
    pts = (pts / pts.new_tensor([[w, h]]) * 2 - 1)[:, None]
    # @TODO: This might still be a source of noise --> bilinear interpolation dangerous
    interp_lin = grid_sample(fmap, pts, align_corners=False, mode="bilinear")
    interp_nn = grid_sample(fmap, pts, align_corners=False, mode="nearest")
    return torch.where(torch.isnan(interp_lin), interp_nn, interp_lin)[:, :, 0].permute(
        0, 2, 1
    )


def sample_depth(pts, depth_):
    depth = torch.where(depth_ > 0, depth_, depth_.new_tensor(float("nan")))
    depth = depth[:, None]
    interp = sample_fmap(pts, depth).squeeze(-1)
    valid = (~torch.isnan(interp)) & (interp > 0)
    return interp, valid


def sample_normals_from_depth(pts, depth, K):
    depth = depth[:, None]
    normals = kornia.geometry.depth.depth_to_normals(depth, K)
    normals = torch.where(depth > 0, normals, 0.0)
    interp = sample_fmap(pts, normals)
    valid = (~torch.isnan(interp)) & (interp > 0)
    return interp, valid


def project(
    kpi,
    di,
    depthj,
    camera_i,
    camera_j,
    T_itoj,
    validi,
    ccth=None,
    sample_depth_fun=sample_depth,
    sample_depth_kwargs=None,
):
    if sample_depth_kwargs is None:
        sample_depth_kwargs = {}

    kpi_3d_i = camera_i.image2cam(kpi)
    kpi_3d_i = kpi_3d_i * di[..., None]
    kpi_3d_j = T_itoj.transform(kpi_3d_i)
    kpi_j, validj = camera_j.cam2image(kpi_3d_j)
    # di_j = kpi_3d_j[..., -1]
    validi = validi & validj
    if depthj is None or ccth is None:
        return kpi_j, validi & validj
    else:
        # circle consistency
        dj, validj = sample_depth_fun(kpi_j, depthj, **sample_depth_kwargs)
        kpi_j_3d_j = camera_j.image2cam(kpi_j) * dj[..., None]
        kpi_j_i, validj_i = camera_i.cam2image(T_itoj.inv().transform(kpi_j_3d_j))
        consistent = ((kpi - kpi_j_i) ** 2).sum(-1) < ccth
        visible = validi & consistent & validj_i & validj
        # visible = validi
        return kpi_j, visible


def dense_warp_consistency(
    depthi: torch.Tensor,
    depthj: torch.Tensor,
    T_itoj: torch.Tensor,
    camerai: Camera,
    cameraj: Camera,
    **kwargs,
):
    kpi = get_image_coords(depthi).flatten(-3, -2)
    di = depthi.flatten(
        -2,
    )
    validi = di > 0
    kpir, validir = project(kpi, di, depthj, camerai, cameraj, T_itoj, validi, **kwargs)

    return kpir.unflatten(-2, depthi.shape[-2:]), validir.unflatten(
        -1, (depthj.shape[-2:])
    )


def unproject(kpts, depth, cam, pose):
    batch, h, w = depth.shape
    in_range = (
        (0 <= kpts[:, :, 0])
        & (kpts[:, :, 0] < w)
        & (0 <= kpts[:, :, 1])
        & (kpts[:, :, 1] < h)
    )
    finite = torch.isfinite(kpts).all(dim=2)
    valid_depth = in_range & finite

    valid_kpts = []
    for b in range(batch):
        valid_kpts.append(kpts[b, valid_depth[b, :], :].to(torch.int64))
    fdepth = torch.full(
        kpts.shape[:2],
        fill_value=float("NaN"),
        device=kpts.device,
        dtype=kpts.dtype,
    )

    for b in range(batch):
        fdepth[b, valid_depth[b]] = depth[b, valid_kpts[b][:, 1], valid_kpts[b][:, 0]]

    kptsc = cam.image2cam(kpts) * fdepth.unsqueeze(-1).repeat(1, 1, 3)
    kptsw = pose.inv().transform(kptsc)

    return kptsw


def simple_project(kptsw, cam, pose):
    ext = pose.transform(kptsw)
    kpts, _ = cam.cam2image(ext)
    return kpts
