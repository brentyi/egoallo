from typing import Tuple
import torch
from torch.nn import functional as F


def get_rot_rep_shape(rot_rep:str) -> Tuple:
    assert rot_rep in ["aa", "quat", "6d", "mat"]
    if rot_rep == "6d":
        return (6,)
    if rot_rep == "aa":
        return (3,)
    if rot_rep == "quat":
        return (4,)
    return (3, 3)


def convert_rotation(rot, src_rep, tgt_rep):
    src_rep, tgt_rep = src_rep.lower(), tgt_rep.lower()
    if src_rep == tgt_rep:
        return rot

    if src_rep == "aa":
        if tgt_rep == "mat":
            return axis_angle_to_matrix(rot)
        if tgt_rep == "quat":
            return axis_angle_to_quaternion(rot)
        if tgt_rep == "6d":
            return axis_angle_to_cont_6d(rot)
        raise NotImplementedError
    if src_rep == "quat":
        if tgt_rep == "aa":
            return quaternion_to_axis_angle(rot)
        if tgt_rep == "mat":
            return quaternion_to_matrix(rot)
        if tgt_rep == "6d":
            return matrix_to_cont_6d(quaternion_to_matrix(rot))
        raise NotImplementedError
    if src_rep == "mat":
        if tgt_rep == "6d":
            return matrix_to_cont_6d(rot)
        if tgt_rep == "aa":
            return matrix_to_axis_angle(rot)
        if tgt_rep == "quat":
            return matrix_to_quaternion(rot)
        raise NotImplementedError
    if src_rep == "6d":
        if tgt_rep == "mat":
            return cont_6d_to_matrix(rot)
        if tgt_rep == "aa":
            return cont_6d_to_axis_angle(rot)
        if tgt_rep == "quat":
            return cont_6d_to_matrix(matrix_to_quaternion(rot))
        raise NotImplementedError
    raise NotImplementedError


def rodrigues_vec_to_matrix(rot_vecs, dtype=torch.float32):
    """
    Calculates the rotation matrices for a batch of rotation vectors
    referenced from https://github.com/mkocabas/VIBE/blob/master/lib/utils/geometry.py
    :param rot_vecs (*, 3) axis-angle vectors
    :returns rot_mats (*, 3, 3)
    """
    dims = rot_vecs.shape[:-1]  # leading dimensions
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=-1, keepdim=True)  # (*, 1)
    rot_dir = rot_vecs / angle  # (*, 3)

    cos = torch.unsqueeze(torch.cos(angle), dim=-2)  # (*, 1, 1)
    sin = torch.unsqueeze(torch.sin(angle), dim=-2)  # (*, 1, 1)

    rx, ry, rz = torch.split(rot_dir, 1, dim=-1)  # (*, 1) each
    zeros = torch.zeros(*dims, 1, dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1).view(
        (*dims, 3, 3)
    )
    I = torch.eye(3, dtype=dtype, device=device).reshape(*(1,) * len(dims), 3, 3)
    rot_mat = I + sin * K + (1 - cos) * torch.einsum("...ij,...jk->...ik", K, K)
    return rot_mat


def matrix_to_axis_angle(matrix):
    """
    Convert rotation matrix to Rodrigues vector
    """
    quaternion = matrix_to_quaternion(matrix)
    aa = quaternion_to_axis_angle(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def axis_angle_to_matrix(rot_vec):
    quaternion = axis_angle_to_quaternion(rot_vec)
    return quaternion_to_matrix(quaternion)


def axis_angle_to_cont_6d(rot_vec):
    """
    :param rot_vec (*, 3)
    :returns 6d vector (*, 6)
    """
    rot_mat = axis_angle_to_matrix(rot_vec)
    return matrix_to_cont_6d(rot_mat)


def matrix_to_cont_6d(matrix):
    """
    :param matrix (*, 3, 3)
    :returns 6d vector (*, 6)
    """
    return torch.cat([matrix[..., 0], matrix[..., 1]], dim=-1)


def cont_6d_to_matrix(cont_6d):
    """
    :param 6d vector (*, 6)
    :returns matrix (*, 3, 3)
    """
    x1 = cont_6d[..., 0:3]
    y1 = cont_6d[..., 3:6]

    x = F.normalize(x1, dim=-1)
    y = F.normalize(y1 - (y1 * x).sum(dim=-1, keepdim=True) * x, dim=-1)
    z = torch.linalg.cross(x, y, dim=-1)

    return torch.stack([x, y, z], dim=-1)


def cont_6d_to_axis_angle(cont_6d):
    rot_mat = cont_6d_to_matrix(cont_6d)
    return matrix_to_axis_angle(rot_mat)


def quaternion_to_axis_angle(quaternion, eps=1e-5):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.
    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    :param quaternion (*, 4) expects WXYZ
    :returns axis_angle (*, 3)
    """
    # unpack input and compute conversion
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]
    sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta = torch.sqrt(sin_squared_theta)
    cos_theta = quaternion[..., 0]
    two_theta = 2.0 * torch.where(
        cos_theta < -eps,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta),
    )

    k_pos = two_theta / sin_theta
    k_neg = 2.0 * torch.ones_like(sin_theta)
    k = torch.where(sin_squared_theta > eps, k_pos, k_neg)

    axis_angle = torch.zeros_like(quaternion)[..., :3]
    axis_angle[..., 0] += q1 * k
    axis_angle[..., 1] += q2 * k
    axis_angle[..., 2] += q3 * k
    return axis_angle


def quaternion_to_matrix(quaternion):
    """
    Convert a quaternion to a rotation matrix.
    Taken from https://github.com/kornia/kornia, based on
    https://github.com/matthew-brett/transforms3d/blob/8965c48401d9e8e66b6a8c37c65f2fc200a076fa/transforms3d/quaternions.py#L101
    https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/geometry/transformation/rotation_matrix_3d.py#L247
    :param quaternion (N, 4) expects WXYZ order
    returns rotation matrix (N, 3, 3)
    """
    # normalize the input quaternion
    quaternion_norm = F.normalize(quaternion, p=2, dim=-1, eps=1e-8)
    *dims, _ = quaternion_norm.shape

    # unpack the normalized quaternion components
    w, x, y, z = torch.chunk(quaternion_norm, chunks=4, dim=-1)

    # compute the actual conversion
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z
    one = torch.tensor(1.0)

    matrix = torch.stack(
        (
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ),
        dim=-1,
    ).view(*dims, 3, 3)
    return matrix


def axis_angle_to_quaternion(axis_angle, eps=1e-5):
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert angle axis to quaternion in WXYZ order
    :param axis_angle (*, 3)
    :returns quaternion (*, 4) WXYZ order
    """
    theta = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    theta_sq = torch.square(theta)
    # theta_sq = torch.sum(axis_angle ** 2, dim=-1, keepdim=True)  # (*, 1)
    # theta = torch.sqrt(theta_sq + eps)
    # need to handle the zero rotation case
    valid = theta_sq > eps
    half_theta = 0.5 * theta
    ones = torch.ones_like(half_theta)
    # fill zero with the limit of sin ax / x -> a
    k = torch.where(valid, torch.sin(half_theta) / (theta + eps), 0.5 * ones)
    w = torch.where(valid, torch.cos(half_theta), ones)
    quat = torch.cat([w, k * axis_angle], dim=-1)
    return quat


def matrix_to_quaternion(matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert rotation matrix to 4d quaternion vector
    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    :param matrix (N, 3, 3)
    """
    *dims, m, n = matrix.shape
    rmat_t = torch.transpose(matrix.reshape(-1, m, n), -1, -2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack(
        [
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            t0,
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
        ],
        -1,
    )
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack(
        [
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            t1,
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
        ],
        -1,
    )
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack(
        [
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
            t2,
        ],
        -1,
    )
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack(
        [
            t3,
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
        ],
        -1,
    )
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(
        t0_rep * mask_c0
        + t1_rep * mask_c1
        + t2_rep * mask_c2  # noqa
        + t3_rep * mask_c3
    )  # noqa
    q *= 0.5
    return q.reshape(*dims, 4)


def quaternion_mul(q0, q1):
    """
    EXPECTS WXYZ
    :param q0 (*, 4)
    :param q1 (*, 4)
    """
    r0, r1 = q0[..., :1], q1[..., :1]
    v0, v1 = q0[..., 1:], q1[..., 1:]
    r = r0 * r1 - (v0 * v1).sum(dim=-1, keepdim=True)
    v = r0 * v1 + r1 * v0 + torch.linalg.cross(v0, v1)
    return torch.cat([r, v], dim=-1)


def quaternion_inverse(q, eps=1e-5):
    """
    EXPECTS WXYZ
    :param q (*, 4)
    """
    conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
    mag = torch.square(q).sum(dim=-1, keepdim=True) + eps
    return conj / mag


def quaternion_slerp(t, q0, q1, eps=1e-5):
    """
    :param t (*, 1)  must be between 0 and 1
    :param q0 (*, 4)
    :param q1 (*, 4)
    """
    dims = q0.shape[:-1]
    t = t.view(*dims, 1)

    q0 = F.normalize(q0, p=2, dim=-1)
    q1 = F.normalize(q1, p=2, dim=-1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)

    # make sure we give the shortest rotation path (< 180d)
    neg = dot < -eps
    q1 = torch.where(neg, -q1, q1)
    dot = torch.where(neg, -dot, dot)
    angle = torch.acos(dot)

    # if angle is too small, just do linear interpolation
    collin = torch.abs(dot) > 1 - eps
    fac = 1 / torch.sin(angle)
    w0 = torch.where(collin, 1 - t, torch.sin((1 - t) * angle) * fac)
    w1 = torch.where(collin, t, torch.sin(t * angle) * fac)
    slerp = q0 * w0 + q1 * w1
    return slerp
