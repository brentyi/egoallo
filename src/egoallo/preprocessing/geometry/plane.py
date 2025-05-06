from jaxtyping import Float
from typing import Tuple, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
from .rotation import axis_angle_to_matrix
from .helpers import make_transform


def transform_align_body_right(root_orient_mat, trans, **kwargs):
    """
    make the transform aligns body_right (-x) with x axis and moves trans to origin
    :param root_orient_mat (3, 3)
    :param trans (3,)
    """
    # move first frame to origin and transform root orient x to [1, 0, 0]
    R_align_x = rotation_align_body_right(root_orient_mat, **kwargs)  # (3, 3)
    t_align_x = -R_align_x @ trans
    return make_transform(R_align_x, t_align_x)


def rotation_align_body_right(
    root_orient_mat, up=[0.0, 0.0, 1.0], right=[-1.0, 0.0, 0.0], **kwargs
):
    """
    compute the rotation that aligns local body right vector (-x)
    with [1, 0, 0] (+x) via rotation about up axis (+z)
    :param root_orient_mat (*, 3, 3)
    :param up vector (*, 3) default [0, 0, 1]
    :param right vector (*, 3) default [1, 0, 0]
    returns (*, 3, 3) rotation matrix
    """
    root_x = -root_orient_mat[..., 0]
    nldims = root_x.ndim - 1
    up = torch.as_tensor(up, device=root_x.device)
    right = torch.as_tensor(right, device=root_x.device)
    if up.ndim < root_x.ndim:
        up = up.reshape(*(1,) * nldims, 3)
    if right.ndim < root_x.ndim:
        right = right.reshape(*(1,) * nldims, 3)

    # project root_x to floor plane (perpendicular to up)
    root_x = root_x - project_vector(root_x, up)
    return rotation_align_vecs(root_x, right)


def compute_world2aligned(T_w0, **kwargs):
    """
    compute alignment transform to take T_w0 to aligned frame
    where body right is -x, and up is as specified (default +z)
    :param T_w0 (*, 4, 4)
    return (*, 4, 4)
    """
    R_aw = rotation_align_body_right(T_w0[..., :3, :3], **kwargs)  # (*..., 3, 3)
    t_aw = torch.einsum("...ij,...j->...i", -R_aw, T_w0[..., :3, 3])
    T_aw = make_transform(R_aw, t_aw)  # (*, 4, 4)
    return T_aw


def rotation_align_vecs(src, target):
    """
    compute rotation taking src to target through the shared plane
    :param src (*, 3)
    :param target (*, 3)
    return (*, 3, 3) rotation matrix
    """
    axis = F.normalize(torch.linalg.cross(src, target), dim=-1)
    angle = torch.arccos(
        (src * target).sum(dim=-1) / (src.norm(dim=-1) * target.norm(dim=-1))
    )
    return axis_angle_to_matrix(axis * angle.unsqueeze(-1))


def compute_point_height(point, floor_plane):
    """
    compute height of point from floor_plane
    :param point (*, 3)
    :param floor_plane (*, 3)
    """
    floor_plane_4d = parse_floor_plane(floor_plane)
    floor_normal = floor_plane_4d[..., :3]
    # compute the distance from root to ground plane
    _, s_root = compute_plane_intersection(point, -floor_normal, floor_plane_4d)
    return s_root


def compute_world2floor(
    floor_plane_4d, root_orient_mat, trans
) -> Tuple[Tensor, Tensor]:
    """
    compute the transform from world frame (opencv +x right, +y down, +z forward),
    to floor frame (-x body right, +y up, with origin at trans)
    :param floor_plane (*, 4) floor plane in world coordinates
    :param root_orient_mat (*, 3, 3) root orientation in world
    :param trans (*, 3) root trans in world
    """
    floor_normal = floor_plane_4d[..., :3]

    # compute prior frame axes in the camera frame
    # right is body +x direction projected to floor plane
    root_x = root_orient_mat[..., 0]
    x = F.normalize(root_x - project_vector(root_x, floor_normal), dim=-1)
    y = floor_normal
    z = F.normalize(torch.linalg.cross(x, y), dim=-1)

    # floor frame in world is body x right, floor normal up
    R_wf = torch.stack([x, y, z], dim=-1)
    R_fw = torch.linalg.inv(R_wf)
    t_fw = torch.einsum("...ij,...j->...i", -R_fw, trans)
    return R_fw, t_fw


def compute_plane_transform(
    plane_4d: Float[Tensor, "*batch 4"],
    up: Float[Tensor, "*batch 3"],
    origin: Optional[Float[Tensor, "*batch 3"]] = None,
):
    """
    compute the R and t transform from identity, where plane normal is up
    """
    normal = plane_4d[..., :3]
    offset = plane_4d[..., 3:]
    normal = F.normalize(normal, dim=-1)
    up = F.normalize(up, dim=-1)
    v = torch.linalg.cross(up, normal)  # (*, 3)
    vnorm = torch.linalg.norm(v, dim=-1, keepdim=True)  # (*, 1)
    s = torch.arcsin(vnorm) / vnorm
    R = axis_angle_to_matrix(v * s)  # (*, 3, 3)
    if origin is not None:
        t, _ = compute_plane_intersection(origin, -normal, plane_4d)
    else:
        # translate plane along normal vector
        t = normal * offset  # (*, 3)
    return R, t


def fit_plane(
    points: Float[Tensor, "*batch N 3"],
    weights: Optional[Float[Tensor, "*batch N 1"]] = None,
    force_sign: int = -1,
) -> Float[Tensor, "*batch 4"]:
    """
    :param points (*, N, 3)
    returns (*, 4) plane parameters (returns in (normal, offset) format)
    """
    *dims, _ = points.shape
    device = points.device
    if weights is None:
        weights = torch.ones(*dims, 1, device=device)

    mean = (weights * points).sum(dim=-2, keepdim=True) / weights.sum(
        dim=-2, keepdim=True
    )
    # (*, N, 3), (*, 3), (*, 3, 3)
    _, _, Vh = torch.linalg.svd(weights * (points - mean))
    normal = Vh[..., -1, :]  # (*, 3)
    offset = torch.einsum("...ij,...j->...i", points, normal)  # (*, N)
    w = weights[..., 0]  # (*, N)
    offset = ((w * offset).sum(dim=-1) / w.sum(dim=-1)).unsqueeze(-1)  # (*, 1)
    if force_sign != 0:
        normal, offset = force_plane_direction(normal, offset, sign=force_sign)
    return torch.cat([normal, offset], dim=-1)


def parse_floor_plane(floor_plane: Tensor, force_sign: int = -1) -> Tensor:
    """
    Takes floor plane in the optimization form (Bx3 with a,b,c * d) and parses into
    (a,b,c,d) from with (a,b,c) normal facing "up in the camera frame and d the offset.
    """
    if floor_plane.shape[-1] == 4:
        return floor_plane

    floor_offset = torch.linalg.norm(floor_plane, dim=-1, keepdim=True)
    floor_normal = floor_plane / (floor_offset + 1e-5)

    # there's ambiguity in the signs of the normal and offset,
    # force the sign of the normal to be positive or negative depending
    # on convention
    if force_sign != 0:
        floor_normal, floor_offset = force_plane_direction(
            floor_normal, floor_offset, sign=force_sign
        )

    return torch.cat([floor_normal, floor_offset], dim=-1)


def force_plane_direction(
    floor_normal: Float[Tensor, "*batch 3"],
    floor_offset: Float[Tensor, "*batch 1"],
    sign: int = -1,
) -> Tuple[Float[Tensor, "*batch 3"], Float[Tensor, "*batch 1"]]:
    assert sign != 0
    if sign > 0:
        mask = floor_normal[..., 1:2] < 0
    else:
        mask = floor_normal[..., 1:2] > 0

    floor_normal = torch.where(
        mask.expand_as(floor_normal), -floor_normal, floor_normal
    )
    floor_offset = torch.where(mask, -floor_offset, floor_offset)
    return floor_normal, floor_offset


def compute_plane_intersection(point, direction, plane, eps=1e-5):
    """
    given a ray defined by a point in space and a direction,
    compute the intersection point with the given plane.
    :param point (*, 3)
    :param direction (*, 3)
    :param plane (*, 4) (normal, offset)
    returns:
        - itsct_pt (*, 3)
        - s (*, 1) s.t. itsct_pt = point + s * direction
    """
    plane_normal = plane[..., :3]
    plane_off = plane[..., 3:]
    s = (plane_off - bdot(plane_normal, point)) / (bdot(plane_normal, direction) + eps)
    itsct_pt = point + s * direction
    return itsct_pt, s


def project_vector(x, d):
    """
    project x onto d
    :param x, d (*, 3)
    """
    d = F.normalize(d, dim=-1)
    return bdot(x, d) * d


def bdot(A1, A2, keepdim=True, **kwargs):
    """
    batched dot product
    :param A1, A2 (*, D)
    returs (*, 1)
    """
    return (A1 * A2).sum(dim=-1, keepdim=keepdim, **kwargs)
