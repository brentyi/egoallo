from jaxtyping import Float, Int
from typing import Tuple, Optional
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch import Tensor

from ..geometry import (
    get_rot_rep_shape,
    convert_rotation,
    batch_apply_Rt,
    make_transform,
    transform_rel_to_global,
    transform_global_to_rel,
)
from .specs import SMPL_JOINTS, smpl_to_openpose, POSE_REFLECT_PERM
from .skeleton import joint_angles_glob_to_rel


__all__ = [
    "run_smpl",
    "reflect_pose_aa",
    "reflect_root_trajectory",
    "forward_kinematics",
    "inverse_kinematics",
    "smpl_local_to_global",
    "select_smpl_joints",
    "get_openpose_from_smpl",
    "convert_local_pose_to_aa",
    "convert_global_pose_to_aa",
    "load_beta_conversion",
    "convert_model_betas",
]


def run_smpl(
    body_model, mats_in: bool = False, return_verts: bool = True, **kwargs
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """
    helper function for running SMPL model with multiple leading dimensions
    return joints and optionally verts and faces
    :param body_model
    :param return_verts (optional bool=True)
    """
    device = body_model.bm.shapedirs.device
    dims = (body_model.batch_size,)
    fields = ["pose_body", "root_orient", "trans", "betas"]
    dim_idcs = [-3 if mats_in else -1, -2 if mats_in else -1, -1, -1]
    for name, idx in zip(fields, dim_idcs):
        if name in kwargs:
            x = kwargs[name]
            if x is None:
                continue
            dims, sh = x.shape[:idx], x.shape[idx:]
            kwargs[name] = x.reshape(-1, *sh).to(device)

    if not return_verts:
        joints = body_model.forward_joints(**kwargs).reshape(*dims, -1, 3)
        return joints, None, None

    smpl = body_model(**kwargs)
    joints = smpl.Jtr.reshape(*dims, -1, 3)
    verts = smpl.v.reshape(*dims, -1, 3)
    return joints, verts, smpl.f


def reflect_pose_aa(root_orient: Tensor, pose_body: Tensor):
    """
    :param root_orient (*, 3)
    :param pose_body (*, (J-1)*3)
    return reflected root_orient and pose_body
    """
    pose_full = torch.cat([root_orient, pose_body], dim=-1)  # (*, J*3)
    pose_reflect = pose_full[..., POSE_REFLECT_PERM]
    pose_reflect[..., 1::3] = -pose_reflect[..., 1::3]
    pose_reflect[..., 2::3] = -pose_reflect[..., 2::3]
    return pose_reflect[..., :3], pose_reflect[..., 3:]


def reflect_root_trajectory(
    rot_aa: Tensor, trans: Tensor, rot_aa_r: Tensor, root_loc: Tensor
) -> Tuple[Tensor, Tensor]:
    # rotation from t to world
    R_wt = convert_rotation(rot_aa, "aa", "mat")
    # get the transforms of the root in the world
    T_wt = make_transform(R_wt, trans + root_loc)
    # transform from t to previous
    T_pt = transform_global_to_rel(T_wt)
    # rotation from reflected t to world
    R_wtr = convert_rotation(rot_aa_r, "aa", "mat")
    # relative transforms
    R_prtr = transform_global_to_rel(R_wtr)
    # get the displacement between t and t-1 in t, the SOURCE frame
    # t_prt = R_prtr * t_tr, where t_tr = (-1, 1, 1) * t_t, and t_t = R_tp * t_pt
    t_tt = torch.einsum("tij,tj->ti", torch.linalg.inv(T_pt[:, :3, :3]), T_pt[:, :3, 3])
    # reflect through x
    t_trtr = torch.cat([-t_tt[..., :1], t_tt[..., 1:]], dim=-1)
    # convert displacement into the t-1 TARGET frame
    t_prtr = torch.einsum("tij,tj->ti", R_prtr, t_trtr)
    # get back global trajectory
    T_prtr = make_transform(R_prtr, t_prtr)
    T_wtr = transform_rel_to_global(T_prtr)
    # get back the smpl translation
    trans_wtr = T_wtr[:, :3, 3] - root_loc
    return rot_aa_r, trans_wtr


def forward_kinematics(
    rot_mats: Tensor,
    joints_in: Tensor,
    parents: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    get the forward transformed joints
    very similar to smplx's batch_rigid_transform with more flexible batch dimensions
    :param rot_mats (*, J, 3, 3) joint rotations from joint i to parent
    :param joints_in (*, J, 3)
    :param parents (J)
    returns (*, J, 4, 4) tensor of transforms
    """
    J = len(parents)
    joints_body_rel = (
        joints_in[..., 1:J, :] - joints_in[..., parents[1:], :]
    )  # (*, J-1, 3)
    joints_rel = torch.cat(
        [joints_in[..., :1, :], joints_body_rel], dim=-2
    )  # (*, J, 3)
    T_pi = make_transform(rot_mats, joints_rel)  # (*, J, 4, 4)
    tforms_wp = [T_pi[..., 0, :, :]]
    for i in range(1, J):
        tforms_wp.append(torch.matmul(tforms_wp[parents[i]], T_pi[..., i, :, :]))
    transforms = torch.stack(tforms_wp, dim=-3)
    joints_posed = transforms[..., :3, 3]  # (*, J, 3)
    rel_trans_h = F.pad(
        joints_posed
        - torch.einsum("...ij,...j->...i", transforms[..., :3, :3], joints_in),
        [0, 1],
        value=1.0,
    ).unsqueeze(-1)
    rel_transforms = torch.cat([transforms[..., :3], rel_trans_h], dim=-1)
    return joints_posed, rel_transforms


def get_pose_offsets(
    pose_mats: Float[Tensor, "*batch J 3 3"],
    posedirs: Float[Tensor, "P N"],
) -> Float[Tensor, "*batch J 3"]:
    dims = pose_mats.shape[:-3]
    I = torch.eye(3, device=pose_mats.device).reshape(*(1,) * len(dims), 1, 3, 3)
    pose_feat = (pose_mats[..., 1:, :, :] - I).reshape(*dims, -1)  # (*, P)
    return torch.einsum("...p,pn->...n", pose_feat, posedirs).reshape(*dims, -1, 3)


def select_vert_params(
    idcs: Int[Tensor, "S"],
    v_template: Float[Tensor, "V 3"],
    shapedirs: Float[Tensor, "V 3 B"],
    posedirs: Float[Tensor, "P N"],
    lbs_weights: Float[Tensor, "V J"],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    pose_idcs = torch.repeat_interleave(3 * idcs, 3, -1)
    pose_idcs[1::3] += 1
    pose_idcs[2::3] += 2
    return v_template[idcs], shapedirs[idcs], posedirs[:, pose_idcs], lbs_weights[idcs]


def get_verts_with_transforms(
    betas: Float[Tensor, "*batch B"],
    pose_mats: Float[Tensor, "*batch J 3 3"],
    rel_transforms: Float[Tensor, "*batch J 4 4"],
    v_template: Float[Tensor, "V 3"],
    shapedirs: Float[Tensor, "V 3 B"],
    posedirs: Float[Tensor, "P N"],
    lbs_weights: Float[Tensor, "V J"],
):
    # (*, V, 3)
    v_shaped = v_template + torch.einsum("...l,mkl->...mk", betas, shapedirs)
    v_posed = v_shaped + get_pose_offsets(pose_mats, posedirs)
    T = torch.einsum("ij,...jkl->...ikl", lbs_weights, rel_transforms)  # (*, V, 4, 4)
    v_out = torch.einsum("...ij,...j->...i", T[..., :3, :3], v_posed) + T[..., :3, 3]
    return v_out


def inverse_kinematics(rot_mats: Tensor, joints: Tensor, parents: Tensor) -> Tensor:
    """
    given the joint rotations and locations of a posed skeleton,
    invert and get the template skeleton
    :param rot_mats (*, J, 3, 3) rotation from joint i to parent (R_pi)
    :param joints (*, J, 3) posed joint locations
    :param parents (J)
    returns (*, J, 3) template joints
    """
    # J = len(parents)
    J = joints.shape[-2]
    # delta between joint and parent in the world
    delta_w = torch.cat(
        [joints[..., :1, :], joints[..., 1:J, :] - joints[..., parents[1:J], :]], dim=-2
    )
    # rot mats from parent to joint i
    rots_ip = rot_mats.transpose(-1, -2)
    # get the world to parent rotation matrices
    rots_pw = [rots_ip[..., 0, :, :]]
    trans_p = [joints[..., 0, :]]
    for i in range(1, J):
        # R_iw = R_ip R_pw = (R_pi.T) R_pw
        R_pw = rots_pw[parents[i]]
        delta_p = torch.einsum("...ij,...j->...i", R_pw, delta_w[..., i, :])
        trans_p.append(trans_p[parents[i]] + delta_p)
        if i >= J - 1:
            break
        rots_pw.append(torch.matmul(rots_ip[..., i, :, :], R_pw))
    return torch.stack(trans_p, dim=-2)


def smpl_local_to_global(
    R_root: Tensor, t_root: Tensor, points_l: Tensor, root_l: Tensor
) -> Tensor:
    """
    transform local smpl body to global
    :param T_root (*, 4, 4) root transform from local to world
    :param points_l (*, N, 3) points to transform
    :param root_l (*, 1, 3) root in local coordinates
    """
    return batch_apply_Rt(R_root, t_root, points_l - root_l) + root_l


def select_smpl_joints(joints_full):
    """
    select the first 22 smpl joints from the full joints
    :param joints_full (*, J, 3)
    """
    return joints_full[..., : len(SMPL_JOINTS), :]


def get_openpose_from_smpl(joints_smpl, model_type="smplh"):
    smpl2op_map = smpl_to_openpose(
        model_type,
        use_hands=False,
        use_face=False,
        use_face_contour=False,
        openpose_format="coco25",
    )
    joints3d_op = joints_smpl[..., smpl2op_map, :]
    # hacky way to get hip joints that align with ViTPose keypoints
    # this could be moved elsewhere in the future (and done properly)
    joints3d_op[..., [9, 12], :] = (
        joints3d_op[..., [9, 12], :]
        + 0.25 * (joints3d_op[..., [9, 12], :] - joints3d_op[..., [12, 9], :])
        + 0.5
        * (
            joints3d_op[..., [8], :]
            - 0.5 * (joints3d_op[..., [9, 12], :] + joints3d_op[..., [12, 9], :])
        )
    )
    return joints3d_op


def convert_local_pose_to_aa(pose_body: Tensor, rot_rep: str):
    """
    convert local pose in rotation representation into flattened axis-angle
    :param pose_body (*, J*D)
    :param rot_rep (str)
    returns (*, J*3) flattened aa pose
    """
    if rot_rep == "aa":
        return pose_body
    dims = pose_body.shape[:-1]
    rot_sh = get_rot_rep_shape(rot_rep)
    pose_aa = convert_rotation(
        pose_body.reshape(*dims, -1, *rot_sh), rot_rep, "aa"
    )  # (*, J, 3)
    return pose_aa.reshape(*dims, -1)


def convert_global_pose_to_aa(pose_glob: Tensor, rot_rep: str):
    """
    :param pose_glob (*, J*D)
    :param rot_rep (str)
    returnns (*, J*3) local pose flattened aa
    """
    dims = pose_glob.shape[:-1]
    rot_sh = get_rot_rep_shape(rot_rep)
    pose_glob_mat = convert_rotation(
        pose_glob.reshape(*dims, -1, *rot_sh), rot_rep, "mat"
    )
    pose_rel_mat = joint_angles_glob_to_rel(pose_glob_mat)  # (*, J, 3, 3)
    return convert_rotation(pose_rel_mat, "mat", "aa").reshape(*dims, -1)


def load_beta_conversion(path: str) -> Tuple[Tensor, Tensor]:
    data = np.load(path)
    return torch.from_numpy(data["A"].astype("float32")), torch.from_numpy(
        data["b"].astype("float32")
    )


def convert_model_betas(beta: Tensor, A: Tensor, b: Tensor) -> Tensor:
    """
    :param beta (*, B)
    :param A (B, B)
    :param b (B)
    beta_neutral = A @ beta_gender + b
    """
    *dims, B = beta.shape
    A = A.reshape((*(1,) * len(dims), *A.shape))
    b = b.reshape((*(1,) * len(dims), *b.shape))
    return torch.einsum("...ij,...j->...i", A, beta) + b
