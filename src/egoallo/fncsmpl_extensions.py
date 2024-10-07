"""EgoAllo-specific SMPL utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from . import fncsmpl, transforms


def flip_hand_quats(hand_quats: Float[Tensor, "... 15 4"]) -> Float[Tensor, "... 15 4"]:
    """Convert right hand quaternions to left hand quaternions, or vice versa."""
    return transforms.SO3.exp(
        transforms.SO3(hand_quats).log() * hand_quats.new_tensor([1, -1, -1])
    ).wxyz


def quats_from_6d_rotation(
    rotation_6d: Float[Tensor, "... 6"],
) -> Float[Tensor, "... 4"]:
    """Use Gram-Schmidt to recover rotation matrixes from 6D rotation
    parameterizations, then convert to wxyz quaternions.

    We don't really use 6D rotations in our codebase, so this is just for
    things like interfacing with HAMER."""
    a = rotation_6d[..., :3]
    a /= torch.linalg.norm(a, axis=-1, keepdim=True)
    b = rotation_6d[..., 3:6]
    b -= torch.sum(a * b, dim=-1, keepdim=True) * b
    b /= torch.linalg.norm(b, axis=-1, keepdim=True)
    rotmat = torch.stack(
        [
            a,
            b,
            torch.linalg.cross(a, b),
        ],
        dim=-1,
    ).reshape((16, 3, 3))
    wxyz = transforms.SO3.from_matrix(rotmat).wxyz
    assert wxyz.shape == (*rotation_6d.shape[:-1], 4)
    return wxyz


def get_T_world_cpf(
    mesh: fncsmpl.SmplMesh,
) -> Float[Tensor, "*#batch 7"]:
    """Get the central pupil frame from a mesh. This assumes that we're using the SMPL-H model."""

    assert mesh.verts.shape[-2:] == (6890, 3), "Not using SMPL-H model!"
    right_eye_pos = (mesh.verts[..., 6260, :] + mesh.verts[..., 6262, :]) / 2.0
    left_eye_pos = (mesh.verts[..., 2800, :] + mesh.verts[..., 2802, :]) / 2.0

    # CPF is between the two eyes.
    cpf_pos = (right_eye_pos + left_eye_pos) / 2.0
    # Get orientation from head.
    cpf_orientation = mesh.posed_model.Ts_world_joint[..., 14, :4]

    return torch.cat([cpf_orientation, cpf_pos], dim=-1)


def get_T_head_cpf(shaped: fncsmpl.SmplhShaped) -> Float[Tensor, "*#batch 7"]:
    """Get the central pupil frame with respect to the head (joint 14). This
    assumes that we're using the SMPL-H model."""

    assert shaped.verts_zero.shape[-2:] == (6890, 3), "Not using SMPL-H model!"
    right_eye = (
        shaped.verts_zero[..., 6260, :] + shaped.verts_zero[..., 6262, :]
    ) / 2.0
    left_eye = (shaped.verts_zero[..., 2800, :] + shaped.verts_zero[..., 2802, :]) / 2.0

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


@dataclass(frozen=True)
class ReparameterizedKintree:
    """Working but hacked together helper for reparameterizing kinematic trees.
    API should be revisited to reduce redundant computation...

    This is useful because the parameterization of a kinematic tree (eg, how we define its
    root) has a large impact on optimization dynamics.

    If we place the root at the pelvis of a human and apply a cost that pulls a
    foot forward, this cost will torque the human around the pelvis, pulling
    the head backwards.

    If we place the root at the human's head, on the other hand, a cost that
    pulls the foot forward will pull the pelvis forward.
    """

    orig_parent_indices: tuple[int, ...]
    reparam_parent_indices: tuple[int, ...]
    reparam_invert_local: Bool[Tensor, "joints"]

    # Index mappings.
    reparam_from_orig: Int[Tensor, "joints"]
    orig_from_reparam: Int[Tensor, "joints"]

    reparam_root_parent: int
    """For computing the original root coordinate frame: which reparameterized
    joint it is attached to."""

    def compute_orig_R_parent_joint_and_T_world_root(
        self,
        reparam_T_world_root: Float[Tensor, "*#batch 7"],
        reparam_R_parent_joint: Float[Tensor, "*#batch joints 4"],
        reparam_t_parent_joint: Float[Tensor, "*#batch joints 3"],
    ) -> tuple[Float[Tensor, "*#batch 7"], Float[Tensor, "*#batch joints 4"]]:
        reparam_T_world_joint = fncsmpl.forward_kinematics(
            T_world_root=reparam_T_world_root,
            Rs_parent_joint=reparam_R_parent_joint,
            t_parent_joint=reparam_t_parent_joint,
            parent_indices=self.reparam_parent_indices,
        )

        # Compute absolute joint positions.
        # This is redundant...
        list_ts_world_joint: list[Tensor] = []
        for i in range(self.reparam_root_parent + 1):
            parent = self.reparam_parent_indices[i]
            if parent == -1:
                list_ts_world_joint.append(reparam_t_parent_joint[..., i, :])
            else:
                list_ts_world_joint.append(
                    list_ts_world_joint[parent] + reparam_t_parent_joint[..., i, :]
                )

        orig_T_world_root = transforms.SE3(
            reparam_T_world_joint[..., self.reparam_root_parent, :]
        ) @ transforms.SE3.from_rotation_and_translation(
            transforms.SO3.identity(
                device=reparam_T_world_root.device, dtype=torch.float32
            ),
            -list_ts_world_joint[self.reparam_root_parent],
        )

        orig_R_parent_joint = torch.where(
            self.reparam_invert_local[..., :, None],
            transforms.SO3(reparam_R_parent_joint).inverse().wxyz,
            reparam_R_parent_joint,
        )[..., self.reparam_from_orig, :]
        return orig_T_world_root.wxyz_xyz, orig_R_parent_joint

    def compute_reparam_R_parent_joint(
        self, orig_R_parent_joint: Float[Tensor, "*#batch joints 4"]
    ) -> Float[Tensor, "*#batch joints 4"]:
        reparam_R_parent_joint = orig_R_parent_joint[..., self.orig_from_reparam, :]
        return torch.where(
            self.reparam_invert_local[..., :, None],
            transforms.SO3(reparam_R_parent_joint).inverse().wxyz,
            reparam_R_parent_joint,
        )

    def compute_reparam_t_parent_joint(
        self,
        orig_t_parent_joint: Float[Tensor, "*#batch joints 3"],
    ) -> Float[Tensor, "*#batch  joints 3"]:
        num_joints = len(self.orig_parent_indices)

        # Compute absolute joint positions.
        list_ts_world_joint: list[Tensor] = []
        for i in range(num_joints):
            parent = self.orig_parent_indices[i]
            if parent == -1:
                list_ts_world_joint.append(orig_t_parent_joint[..., i, :])
            else:
                list_ts_world_joint.append(
                    list_ts_world_joint[parent] + orig_t_parent_joint[..., i, :]
                )
        ts_world_joint = torch.stack(list_ts_world_joint, dim=-2)

        # Reorder joint positions.
        reparam_ts_world_joint = ts_world_joint[
            ...,
            self.orig_from_reparam,
            :,
        ]

        # Return positions relative to parent.
        reparam_parent_indices = reparam_ts_world_joint.new_tensor(
            self.reparam_parent_indices, dtype=torch.int32, requires_grad=False
        )
        return reparam_ts_world_joint - torch.where(
            reparam_parent_indices[..., None] >= 0,
            reparam_ts_world_joint[
                ...,
                reparam_parent_indices,
                :,
            ],
            0.0,
        )

    @staticmethod
    def compute(
        orig_parent_indices: tuple[int, ...],
        new_root: int,
        device: torch.device | str,
    ) -> ReparameterizedKintree:
        """Reparameterize a kinematic tree.

        This needs to produce a few things:
        - A new set of parent indices.
        - A boolean mask indicating which local joints need to be inverted.
        - Mapping from orig to reparam.
        - Mapping from reparam to orig.
        - New t_parent_indices.
        """

        num_joints = len(orig_parent_indices)
        orig_children_of = {i: [] for i in range(-1, num_joints)}
        for i, parent in enumerate(orig_parent_indices):
            orig_children_of[parent].append(i)

        orig_visited = set[int]()

        reparam_parent_indices = []
        reparam_from_orig = torch.zeros((num_joints,), dtype=torch.int32, device=device)
        orig_from_reparam = torch.zeros((num_joints,), dtype=torch.int32, device=device)
        reparam_invert_local = torch.zeros(
            (num_joints,), dtype=torch.bool, device=device
        )
        reparam_root_parent = -1

        def dfs(
            reparam_parent: int,
            orig_index: int,
            invert_local: bool,
        ) -> None:
            if orig_index in orig_visited:
                return

            reparam_index = len(reparam_parent_indices)
            orig_visited.add(orig_index)

            if orig_index != -1:
                reparam_parent_indices.append(reparam_parent)
                reparam_from_orig[orig_index] = reparam_index
                orig_from_reparam[reparam_index] = orig_index
                reparam_invert_local[reparam_index] = invert_local

                # Atttempt to visit parent.
                dfs(
                    reparam_parent=reparam_index,
                    orig_index=orig_parent_indices[orig_index],
                    invert_local=True,
                )
            else:
                nonlocal reparam_root_parent
                reparam_root_parent = reparam_parent

            for orig_children in orig_children_of[orig_index]:
                dfs(
                    reparam_parent=reparam_parent if invert_local else reparam_index,
                    orig_index=orig_children,
                    invert_local=False,
                )

        dfs(reparam_parent=-1, orig_index=new_root, invert_local=True)

        return ReparameterizedKintree(
            orig_parent_indices=orig_parent_indices,
            reparam_parent_indices=tuple(reparam_parent_indices),
            reparam_invert_local=reparam_invert_local,
            reparam_from_orig=reparam_from_orig,
            orig_from_reparam=orig_from_reparam,
            reparam_root_parent=reparam_root_parent,
        )
