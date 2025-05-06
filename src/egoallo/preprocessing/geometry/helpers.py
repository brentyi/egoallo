import torch
import numpy as np

from .rotation import convert_rotation


def make_transform(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)


def transform_points(T, x):
    """
    :param T (*, 4, 4)
    :param x (*, N, 3)
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    return batch_apply_Rt(R, t, x)


def batch_apply_Rt(R, t, x):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    :param x (*, N, 3)
    """
    return torch.einsum("...ij,...nj->...ni", R, x) + t.unsqueeze(-2)


def transform_global_to_rel(T_glob):
    """
    get the relative transforms (diffs) from global transform of trajectory
    :param T_glob (*, T, 4, 4) root to world transform
    return root t->t-1 transform (*, T, 4, 4)
    """
    T_rel = torch.matmul(
        torch.linalg.inv(T_glob[..., :-1, :, :]), T_glob[..., 1:, :, :]
    )  # (*, T-1, 4, 4)
    return torch.cat([T_glob[..., :1, :, :], T_rel], dim=-3)


def transform_rel_to_global(T_rel):
    """
    convert relative transforms into global trajectory
    :param T_rel (*, T, 4, 4) root t -> t-1 transform
    return root t -> world transform (*, T, 4, 4)
    """
    N = T_rel.shape[-3]
    T_rel_list = T_rel.unbind(dim=-3)
    T_glob_list = [T_rel_list[0]]
    for t in range(1, N):
        T_cur = torch.matmul(T_glob_list[t - 1], T_rel_list[t])
        T_glob_list.append(T_cur)
    return torch.stack(T_glob_list, dim=-3)


def RT_global_to_rel(R_glob, t_glob):
    """
    :param R_glob (*, T, 3, 3) root to world rotation
    :param t_glob (*, T, 3) root to world translation
    returns root t -> t-1 rotation (*, T, 3, 3) and translation (*, T, 3)
    """
    T_glob = make_transform(R_glob, t_glob)  # (*, T, 4, 4) root to world
    T_rel = transform_global_to_rel(T_glob)
    return T_rel[..., :3, :3], T_rel[..., :3, 3]


def RT_rel_to_global(R_rel, t_rel):
    """
    :param R_rel (*, T, 3, 3) root at t -> root at t-1 rotation
    :param t_rel (*, T, 3) root at t -> root at t-1 translation
    return root to world rotation (*, T, 3, 3) and translation (*, T, 3)
    """
    T_rel = make_transform(R_rel, t_rel)  # (*, T, 4, 4)
    T_glob = transform_rel_to_global(T_rel)
    return T_glob[..., :3, :3], T_glob[..., :3, 3]


def joints_local_to_global(
    root_orient, trans, joints_loc, use_rel: bool = True, rot_rep: str = "6d"
):
    """
    convert joints in local coords to global coordinates
    (X_w - root) = T_wl * (X_l - root)
    :param trans (*, T, 3)
    :param root_orient (*, T, *rot_shape)
    :param joints_loc (*, T, J * 3)
    :param use_rel (optional bool) if true, root trajectory specified as relative transforms
    returns global joint locations (*, T, J, 3)
    """
    root_orient_mat = convert_rotation(root_orient, rot_rep, "mat")  # (B, T, 3, 3)
    T_wl = make_transform(root_orient_mat, trans)
    if use_rel:  # global translation and orientation are in diffs
        T_wl = transform_rel_to_global(T_wl)

    joints_loc = joints_loc.reshape(*trans.shape[:-1], -1, 3)
    root_loc = joints_loc[..., :1, :]
    return transform_points(T_wl, joints_loc - root_loc) + root_loc


def joints_global_to_local(root_orient_mat, trans, joints_glob, joints_vel_glob=None):
    """
    convert joints in global coords to local coords
    i.e. smpl output with zero root_orient and trans
    (X_w - root) = T_wl * (X_l - root)
    :param trans (*, 3)
    :param root_orient_mat (*, 3, 3)
    :param joints_glob (*, J, 3)
    :param joints_vel_glob (optional) (*, J, 3)
    returns local joint locations (*, J, 3)
    """
    T_lw = torch.linalg.inv(make_transform(root_orient_mat, trans))  # (*, 4, 4)
    root_loc = joints_glob[..., :1, :] - trans.unsqueeze(-2)  # (*, 1, 3)
    joints_loc = transform_points(T_lw, joints_glob - root_loc) + root_loc
    joints_vel_loc = None
    if joints_vel_glob is not None:  # no translation
        joints_vel_loc = torch.einsum(
            "...ij,...nj->...ni", T_lw[..., :3, :3], joints_vel_glob
        )  # (*, J, 3)
    return joints_loc, joints_vel_loc


def align_pcl(Y, X, weight=None, fixed_scale=False):
    """
    align similarity transform to align X with Y using umeyama method
    X' = s * R * X + t is aligned with Y
    :param Y (*, N, 3) first trajectory
    :param X (*, N, 3) second trajectory
    :param weight (*, N, 1) optional weight of valid correspondences
    :returns s (*, 1), R (*, 3, 3), t (*, 3)
    """
    *dims, N, _ = Y.shape
    device = X.device
    N = torch.ones(*dims, 1, 1, device=device) * N

    if weight is not None:
        N = weight.sum(dim=-2, keepdim=True)  # (*, 1, 1)

    # subtract mean
    my = Y.sum(dim=-2) / N[..., 0]  # (*, 3)
    mx = X.sum(dim=-2) / N[..., 0]
    y0 = Y - my[..., None, :]  # (*, N, 3)
    x0 = X - mx[..., None, :]

    if weight is not None:
        y0 = y0 * weight
        x0 = x0 * weight

    # correlation
    C = torch.matmul(y0.transpose(-1, -2), x0) / N  # (*, 3, 3)
    U, D, Vh = torch.linalg.svd(C)  # (*, 3, 3), (*, 3), (*, 3, 3)

    S = torch.eye(3, device=device).reshape(*(1,) * (len(dims)), 3, 3).repeat(*dims, 1, 1)
    neg = torch.det(U) * torch.det(Vh.transpose(-1, -2)) < 0
    S[neg, 2, 2] = -1

    R = torch.matmul(U, torch.matmul(S, Vh))  # (*, 3, 3)

    D = torch.diag_embed(D)  # (*, 3, 3)
    if fixed_scale:
        s = torch.ones(*dims, 1, device=device, dtype=torch.float32)
    else:
        var = torch.sum(torch.square(x0), dim=(-1, -2), keepdim=True) / N  # (*, 1, 1)
        s = (
            torch.diagonal(torch.matmul(D, S), dim1=-2, dim2=-1).sum(
                dim=-1, keepdim=True
            )
            / var[..., 0]
        )  # (*, 1)

    t = my - s * torch.matmul(R, mx[..., None])[..., 0]  # (*, 3)

    return s, R, t


def get_translation_scale(fps=30):
    """
    scale relative translation into (m/s), over average walking speed ~1.5 m/s
    i.e. scale delta such that average walking speed -> 1
    """
    return 1.0 * fps


def estimate_velocity(data_seq, h=1 / 30):
    """
    Given some data sequence of T timesteps in the shape (T, ...), estimates
    the velocity for the middle T-2 steps using a second order central difference scheme.
    - h : step size
    """
    data_tp1 = data_seq[2:]
    data_tm1 = data_seq[0:-2]
    data_vel_seq = (data_tp1 - data_tm1) / (2 * h)
    return data_vel_seq


def estimate_angular_velocity(rot_seq, h=1 / 30):
    """
    Given a sequence of T rotation matrices, estimates angular velocity at T-2 steps.
    Input sequence should be of shape (T, ..., 3, 3)
    """
    # see https://en.wikipedia.org/wiki/Angular_velocity#Calculation_from_the_orientation_matrix
    dRdt = estimate_velocity(rot_seq, h)
    R = rot_seq[1:-1]
    RT = np.swapaxes(R, -1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = np.matmul(dRdt, RT)

    # pull out angular velocity vector
    # average symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = np.stack([w_x, w_y, w_z], axis=-1)
    return w
