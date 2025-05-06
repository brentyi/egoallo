from typing import Tuple
import torch
import numpy as np

# import lietorch as tf
from . import transforms as tf
from .helpers import batch_apply_Rt


def project_from_world(X_w, R_cw, t_cw, intrins):
    """
    :param X_w (*, N, 3)
    :param cam_R (*, 3, 3)
    :param cam_t (*, 3)
    :param intrins (*, 4)
    """
    return proj_2d(batch_apply_Rt(R_cw, t_cw, X_w), intrins)


def proj_2d(xyz, intrins, eps=1e-4):
    """
    :param xyz (*, 3/4) 3d/4d point in camera coordinates
    :param intrins (*, 4) fx, fy, cx, cy
    return (*, 2) of reprojected points, (*) of points in front of camera
    """
    z = xyz[..., 2:3]
    valid_mask = z > eps
    disp = torch.where(valid_mask, 1.0 / (z + eps), torch.ones_like(z))
    focal = intrins[..., :2]
    center = intrins[..., 2:]
    return focal * disp * xyz[..., :2] + center, valid_mask[..., 0]


def proj_h(xyzw):
    """
    project homogeneous point
    """
    w = xyzw[..., -1:]
    return xyzw[..., :-1] * torch.where(w > 0, 1.0 / w, w)


def iproj_depth(uv, z, intrins):
    """
    inverse project into 3d coords from depth
    :param uv (*, 2)
    :param z (*, 1)
    :param intrins (*, 4)
    :returns (*, 3)
    """
    focal = intrins[..., :2]
    center = intrins[..., 2:]
    return z * torch.cat([(uv - center) / focal, torch.ones_like(z)], dim=-1)


def iproj(uv, disp, intrins):
    """
    inverse project from disparity. returns 4d homogeneous
    :param uv (*, 2)
    :param disp (*, 1)
    :param intrins (*, 4)
    :returns (*, 4)
    """
    x = normalize_coords(uv, intrins)
    X = torch.cat([x, torch.ones_like(disp), disp], dim=-1)
    return X


def normalize_coords(uv, intrins):
    focal = intrins[..., :2]
    center = intrins[..., 2:]
    return (uv - center) / focal


def iproj_to_world(uv, disp, intrins, extrins, ret_3d=True):
    """
    inverse project disparity into world coords. default returns 3d
    :param uv (*, 2)
    :param disp (*, 1)
    :param intrins (*, 4)
    :param extrins (*, 7)
    :param ret_3d (optional bool) return in 3d coords, default True
    :returns (*, 3)
    """
    T_wc = tf.SE3(extrins).inv()
    X_c = iproj(uv, disp, intrins)
    X_w = T_wc.act(X_c)
    if ret_3d:
        return proj_h(X_w)
    return X_w


def reproject(pose_params, intrins, disps, uv, ii, jj):
    """
    :param pose_params (T, *, 7) pose parameters
    :param intrins (T, *, 4) fx, fy, cx, cy
    :param uv (T, *, 2) coordinate grid
    :param disps (T, *, 1) disparity
    :param ii (N) source index array into parameters
    :param jj (N) target index array into parameters
    returns (N, *, 2) points in ii reprojected into jj
    """
    T_i, T_j = tf.SE3(pose_params[ii]), tf.SE3(pose_params[jj])
    Xh_i = iproj(uv, disps[ii], intrins[ii])
    Xh_j = T_j.mul(T_i.inv()).act(Xh_i)
    return proj_2d(Xh_j, intrins[jj])


def proj_2d_jac(X, intrins):
    """
    :param X (*, 4) point in camera coordinates
    :param intrins (*, 4) fx, fy, cx, cy
    return (*, 2, 4)
    """
    fx, fy, cx, cy = intrins.unbind(dim=-1)
    X, Y, Z, D = X.unbind(dim=-1)
    d = torch.where(Z > 0.1, 1.0 / Z, torch.ones_like(Z))
    o = torch.zeros_like(d)
    return torch.stack(
        [fx * d, o, -fx * X * d * d, o, o, fy * d, -fy * Y * d * d, o],
        dim=-1,
    ).reshape(*d.shape, 2, 4)


def actp_jac(X1):
    """
    :param X1 (*, 4) point after transformation
    """
    x, y, z, d = X1.unbind(dim=-1)
    o = torch.zeros_like(d)
    return torch.stack(
        [d, o, o, o, z, -y, o, d, o, -z, o, x, o, o, d, y, -x, o, o, o, o, o, o, o],
        dim=-1,
    ).reshape(*d.shape, 4, 6)


def iproj_jac(X):
    """
    jacobian for inverse projection to 4d
    """
    J = torch.zeros_like(X)
    J[..., -1] = 1
    return J


def make_homogeneous(x):
    """
    :param x (*, 3)
    returns x in homogeneous coordinates
    """
    return torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)


def make_transform(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    bottom = (
        torch.tensor([0, 0, 0, 1], dtype=R.dtype, device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .repeat(*dims, 1, 1)
    )
    return torch.cat([torch.cat([R, t.unsqueeze(-1)], dim=-1), bottom], dim=-2)


def focal2fov(focal, R):
    """
    :param focal, focal length
    :param R, either W / 2 or H / 2
    """
    return 2 * np.arctan(R / focal)


def fov2focal(fov, R):
    """
    :param fov, field of view in radians
    :param R, either W / 2 or H / 2
    """
    return R / np.tan(fov / 2)


def lookat_matrix(source_pos, target_pos, up):
    """
    uses x right y down z forward opencv convention
    :param source_pos (*, 3)
    :param target_pos (*, 3)
    :param up (3,)
    """
    *dims, _ = source_pos.shape
    up = up.reshape(*(1,) * len(dims), 3)
    up = up / torch.linalg.norm(up, dim=-1, keepdim=True)
    back = normalize(source_pos - target_pos)
    right = normalize(torch.linalg.cross(up, back))
    up = normalize(torch.linalg.cross(back, right))
    R = torch.stack([right, -up, -back], dim=-1)
    return make_transform(R, source_pos)


def normalize(x):
    return x / torch.linalg.norm(x, dim=-1, keepdim=True)


def view_matrix(z, up, pos):
    """
    :param z (*, 3) up (*, 3) pos (*, 3)
    returns (*, 4, 4)
    """
    *dims, _ = z.shape
    x = normalize(torch.linalg.cross(up, z))
    y = normalize(torch.linalg.cross(z, x))
    bottom = (
        torch.tensor([0, 0, 0, 1], dtype=torch.float32)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )

    return torch.cat([torch.stack([x, y, z, pos], dim=-1), bottom], dim=-2)


def average_pose(poses):
    """
    :param poses (N, 4, 4)
    returns average pose (4, 4)
    """
    center = poses[:, :3, 3].mean(0)
    up = normalize(poses[:, :3, 1].sum(0))
    z = normalize(poses[:, :3, 2].sum(0))
    return view_matrix(z, up, center)


def make_translation(t):
    return make_transform(torch.eye(3, device=t.device), t)


def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    else:
        raise NotImplementedError
    return make_transform(R, torch.zeros(3))


def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )


def identity(shape: Tuple, d=4, **kwargs):
    I = torch.eye(d, **kwargs)
    return I.reshape(*(1,) * len(shape), d, d).repeat(*shape, 1, 1)
