"""EgoAllo-specific SMPL utilities."""

from __future__ import annotations


import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from . import fncsmpl, transforms


def get_T_world_cpf(mesh: fncsmpl.SmplMesh) -> Float[Tensor, "*#batch 7"]:
    """Get the central pupil frame from a mesh. This assumes that we're using the SMPL-H model."""

    assert mesh.verts.shape[-2:] == (6890, 3), "Not using SMPL-H model!"
    right_eye = (mesh.verts[..., 6260, :] + mesh.verts[..., 6262, :]) / 2.0
    left_eye = (mesh.verts[..., 2800, :] + mesh.verts[..., 2802, :]) / 2.0

    # CPF is between the two eyes.
    cpf_pos = (right_eye + left_eye) / 2.0
    # Get orientation from head.
    cpf_orientation = mesh.posed_model.Ts_world_joint[..., 14, :4]

    return torch.cat([cpf_orientation, cpf_pos], dim=-1)


def get_T_head_cpf(shaped: fncsmpl.SmplhShaped) -> Float[Tensor, "*#batch 7"]:
    """Get the central pupil frame with respect to the head (joint 14). This
    assumes that we're using the SMPL-H model."""

    verts_zero = shaped.verts_zero

    assert verts_zero.shape[-2:] == (6890, 3), "Not using SMPL-H model!"
    right_eye = (verts_zero[..., 6260, :] + verts_zero[..., 6262, :]) / 2.0
    left_eye = (verts_zero[..., 2800, :] + verts_zero[..., 2802, :]) / 2.0

    # CPF is between the two eyes.
    cpf_pos_wrt_head = (right_eye + left_eye) / 2.0 - shaped.joints_zero[..., 14, :]

    return fncsmpl.broadcasting_cat(
        [
            transforms.SO3.identity(
                device=cpf_pos_wrt_head.device, dtype=cpf_pos_wrt_head.dtype
            ).wxyz,
            cpf_pos_wrt_head,
        ],
        dim=-1,
    )


def get_T_world_root_from_cpf_pose(
    posed: fncsmpl.SmplhShapedAndPosed,
    Ts_world_cpf: Float[Tensor | np.ndarray, "... 7"],
) -> Float[Tensor, "... 7"]:
    """Get the root transform that would align the CPF frame of `posed` to `Ts_world_cpf`."""
    device = posed.Ts_world_joint.device
    dtype = posed.Ts_world_joint.dtype

    if isinstance(Ts_world_cpf, np.ndarray):
        Ts_world_cpf = torch.from_numpy(Ts_world_cpf).to(device=device, dtype=dtype)

    assert Ts_world_cpf.shape[-1] == 7
    T_world_root = (
        # T_world_cpf
        transforms.SE3(Ts_world_cpf)
        # T_cpf_head
        @ transforms.SE3(get_T_head_cpf(posed.shaped_model)).inverse()
        # T_head_world
        @ transforms.SE3(posed.Ts_world_joint[..., 14, :]).inverse()
        # T_world_root
        @ transforms.SE3(posed.T_world_root)
    )
    return T_world_root.wxyz_xyz
