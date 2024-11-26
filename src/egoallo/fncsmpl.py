"""Somewhat opinionated wrapper for the SMPL-H body model.

Very little of it is specific to SMPL-H. This could very easily be adapted for other models in SMPL family.

We break down the SMPL-H into four stages, each with a corresponding data structure:
- Loading the model itself:
    `model = SmplhModel.load(path to npz)`
- Applying a body shape to the model:
    `shaped = model.with_shape(betas)`
- Posing the body shape:
    `posed = shaped.with_pose(root pose, local joint poses)`
- Recovering the mesh with LBS:
    `mesh = posed.lbs()`

In contrast to other SMPL wrappers:
- Everything is stateless, so we can support arbitrary batch axes.
- The root is no longer ever called a joint.
- The `trans` and `root_orient` inputs are replaced by a single SE(3) root transformation.
- We're using (4,) wxyz quaternion vectors for all rotations, (7,) wxyz_xyz vectors for all
  rigid transforms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from einops import einsum
from jaxtyping import Float, Int
from torch import Tensor

from .tensor_dataclass import TensorDataclass
from .transforms import SE3, SO3


class SmplhModel(TensorDataclass):
    """A human body model from the SMPL family."""

    faces: Int[Tensor, "faces 3"]
    """Vertex indices for mesh faces."""
    J_regressor: Float[Tensor, "joints+1 verts"]
    """Linear map from vertex to joint positions.
    For SMPL-H, 1 root + 21 body joints + 2 * 15 hand joints."""
    parent_indices: tuple[int, ...]
    """Defines kinematic tree. Index of -1 signifies that a joint is defined
    relative to the root."""
    weights: Float[Tensor, "verts joints+1"]
    """LBS weights."""
    posedirs: Float[Tensor, "verts 3 joints*9"]
    """Pose blend shape bases."""
    v_template: Float[Tensor, "verts 3"]
    """Canonical mesh verts."""
    shapedirs: Float[Tensor, "verts 3 n_betas"]
    """Shape bases."""

    @staticmethod
    def load(model_path: Path) -> SmplhModel:
        """Load a body model from an NPZ file."""
        params_numpy: dict[str, np.ndarray] = {
            k: _normalize_dtype(v)
            for k, v in np.load(model_path, allow_pickle=True).items()
        }
        assert (
            "bs_style" not in params_numpy
            or params_numpy.pop("bs_style").item() == b"lbs"
        )
        assert (
            "bs_type" not in params_numpy
            or params_numpy.pop("bs_type").item() == b"lrotmin"
        )
        parent_indices = tuple(
            int(index) for index in params_numpy.pop("kintree_table")[0][1:] - 1
        )
        params = {
            k: torch.from_numpy(v)
            for k, v in params_numpy.items()
            if v.dtype in (np.int32, np.float32)
        }
        return SmplhModel(
            faces=params["f"],
            J_regressor=params["J_regressor"],
            parent_indices=parent_indices,
            weights=params["weights"],
            posedirs=params["posedirs"],
            v_template=params["v_template"],
            shapedirs=params["shapedirs"],
        )

    def get_num_joints(self) -> int:
        """Get the number of joints in this model."""
        return len(self.parent_indices)

    def with_shape(self, betas: Float[Tensor, "*#batch n_betas"]) -> SmplhShaped:
        """Compute a new body model, with betas applied."""
        num_betas = betas.shape[-1]
        assert num_betas <= self.shapedirs.shape[-1]
        verts_with_shape = self.v_template + einsum(
            self.shapedirs[:, :, :num_betas],
            betas,
            "verts xyz beta, ... beta -> ... verts xyz",
        )
        root_and_joints_pred = einsum(
            self.J_regressor,
            verts_with_shape,
            "jointsp1 verts, ... verts xyz -> ... jointsp1 xyz",
        )
        root_offset = root_and_joints_pred[..., 0:1, :]
        return SmplhShaped(
            body_model=self,
            root_offset=root_offset.unsqueeze(-2),
            verts_zero=verts_with_shape - root_offset,
            joints_zero=root_and_joints_pred[..., 1:, :] - root_offset,
            t_parent_joint=root_and_joints_pred[..., 1:, :]
            - root_and_joints_pred[..., np.array(self.parent_indices) + 1, :],
        )


class SmplhShaped(TensorDataclass):
    """The SMPL-H body model with a body shape applied."""

    body_model: SmplhModel
    """The underlying body model."""
    root_offset: Float[Tensor, "*#batch 3"]
    verts_zero: Float[Tensor, "*#batch verts 3"]
    """Vertices of shaped body _relative to the root joint_ at the zero
    configuration."""
    joints_zero: Float[Tensor, "*#batch joints 3"]
    """Joints of shaped body _relative to the root joint_ at the zero
    configuration."""
    t_parent_joint: Float[Tensor, "*#batch joints 3"]
    """Position of each shaped body joint relative to its parent. Does not
    include root."""

    def with_pose_decomposed(
        self,
        T_world_root: Float[Tensor, "*#batch 7"],
        body_quats: Float[Tensor, "*#batch 21 4"],
        left_hand_quats: Float[Tensor, "*#batch 15 4"] | None = None,
        right_hand_quats: Float[Tensor, "*#batch 15 4"] | None = None,
    ) -> SmplhShapedAndPosed:
        """Pose our SMPL-H body model. Returns a set of joint and vertex outputs."""

        num_joints = self.body_model.get_num_joints()
        batch_axes = body_quats.shape[:-2]
        if left_hand_quats is None:
            left_hand_quats = body_quats.new_zeros((*batch_axes, 15, 4))
            left_hand_quats[..., 0] = 1.0
        if right_hand_quats is None:
            right_hand_quats = body_quats.new_zeros((*batch_axes, 15, 4))
            right_hand_quats[..., 0] = 1.0
        local_quats = broadcasting_cat(
            [body_quats, left_hand_quats, right_hand_quats], dim=-2
        )
        assert local_quats.shape[-2:] == (num_joints, 4)
        return self.with_pose(T_world_root, local_quats)

    def with_pose(
        self,
        T_world_root: Float[Tensor, "*#batch 7"],
        local_quats: Float[Tensor, "*#batch joints 4"],
    ) -> SmplhShapedAndPosed:
        """Pose our SMPL-H body model. Returns a set of joint and vertex outputs."""

        # Forward kinematics.
        num_joints = self.body_model.get_num_joints()
        assert local_quats.shape[-2:] == (num_joints, 4)
        Ts_world_joint = forward_kinematics(
            T_world_root=T_world_root,
            Rs_parent_joint=local_quats,
            t_parent_joint=self.t_parent_joint,
            parent_indices=self.body_model.parent_indices,
        )
        assert Ts_world_joint.shape[-2:] == (num_joints, 7)
        return SmplhShapedAndPosed(
            shaped_model=self,
            T_world_root=T_world_root,
            local_quats=local_quats,
            Ts_world_joint=Ts_world_joint,
        )


class SmplhShapedAndPosed(TensorDataclass):
    shaped_model: SmplhShaped
    """Underlying shaped body model."""

    T_world_root: Float[Tensor, "*#batch 7"]
    """Root coordinate frame."""

    local_quats: Float[Tensor, "*#batch joints 4"]
    """Local joint orientations."""

    Ts_world_joint: Float[Tensor, "*#batch joints 7"]
    """Absolute transform for each joint. Does not include the root."""

    def with_new_T_world_root(
        self, T_world_root: Float[Tensor, "*#batch 7"]
    ) -> SmplhShapedAndPosed:
        return SmplhShapedAndPosed(
            shaped_model=self.shaped_model,
            T_world_root=T_world_root,
            local_quats=self.local_quats,
            Ts_world_joint=(
                SE3(T_world_root[..., None, :])
                @ SE3(self.T_world_root[..., None, :]).inverse()
                @ SE3(self.Ts_world_joint)
            ).parameters(),
        )

    def lbs(self) -> SmplMesh:
        """Compute a mesh with LBS."""
        num_joints = self.local_quats.shape[-2]
        verts_with_blend = self.shaped_model.verts_zero + einsum(
            self.shaped_model.body_model.posedirs,
            (
                SO3(self.local_quats).as_matrix()
                - torch.eye(
                    3, dtype=self.local_quats.dtype, device=self.local_quats.device
                )
            ).reshape((*self.local_quats.shape[:-2], num_joints * 9)),
            "... verts j joints_times_9, ... joints_times_9 -> ... verts j",
        )
        verts_transformed = einsum(
            broadcasting_cat(
                [
                    SE3(self.T_world_root).as_matrix()[..., None, :3, :],
                    SE3(self.Ts_world_joint).as_matrix()[..., :, :3, :],
                ],
                dim=-3,
            ),
            self.shaped_model.body_model.weights,
            broadcasting_cat(
                [
                    verts_with_blend[..., :, None, :]
                    - broadcasting_cat(  # Prepend root to joints zeros.
                        [
                            self.shaped_model.joints_zero.new_zeros(3),
                            self.shaped_model.joints_zero[..., None, :, :],
                        ],
                        dim=-2,
                    ),
                    verts_with_blend.new_ones(
                        (
                            *verts_with_blend.shape[:-1],
                            1 + self.shaped_model.joints_zero.shape[-2],
                            1,
                        )
                    ),
                ],
                dim=-1,
            ),
            "... joints_p1 i j, verts joints_p1, ... verts joints_p1 j -> ... verts i",
        )
        assert (
            verts_transformed.shape[-2:]
            == self.shaped_model.body_model.v_template.shape
        )
        return SmplMesh(
            posed_model=self,
            verts=verts_transformed,
            faces=self.shaped_model.body_model.faces,
        )


class SmplMesh(TensorDataclass):
    """Outputs from the SMPL-H model."""

    posed_model: SmplhShapedAndPosed
    """Posed model that this mesh was computed for."""

    verts: Float[Tensor, "*#batch verts 3"]
    """Vertices for mesh."""

    faces: Int[Tensor, "verts 3"]
    """Faces for mesh."""


def forward_kinematics(
    T_world_root: Float[Tensor, "*#batch 7"],
    Rs_parent_joint: Float[Tensor, "*#batch joints 4"],
    t_parent_joint: Float[Tensor, "*#batch joints 3"],
    parent_indices: tuple[int, ...],
) -> Float[Tensor, "*#batch joints 7"]:
    """Run forward kinematics to compute absolute poses (T_world_joint) for
    each joint. The output array containts pose parameters
    (w, x, y, z, tx, ty, tz) for each joint. (this does not include the root!)

    Args:
        T_world_root: Transformation to world frame from root frame.
        Rs_parent_joint: Local orientation of each joint.
        t_parent_joint: Position of each joint with respect to its parent frame. (this does not
            depend on local joint orientations)
        parent_indices: Parent index for each joint. Index of -1 signifies that
            a joint is defined relative to the root. We assume that this array is
            sorted: parent joints should always precede child joints.

    Returns:
        Transformations to world frame from each joint frame.
    """

    # Check shapes.
    num_joints = len(parent_indices)
    assert Rs_parent_joint.shape[-2:] == (num_joints, 4)
    assert t_parent_joint.shape[-2:] == (num_joints, 3)

    # Get relative transforms.
    Ts_parent_child = broadcasting_cat([Rs_parent_joint, t_parent_joint], dim=-1)
    assert Ts_parent_child.shape[-2:] == (num_joints, 7)

    # Compute one joint at a time.
    list_Ts_world_joint: list[Tensor] = []
    for i in range(num_joints):
        if parent_indices[i] == -1:
            T_world_parent = T_world_root
        else:
            T_world_parent = list_Ts_world_joint[parent_indices[i]]
        list_Ts_world_joint.append(
            (SE3(T_world_parent) @ SE3(Ts_parent_child[..., i, :])).wxyz_xyz
        )

    Ts_world_joint = torch.stack(list_Ts_world_joint, dim=-2)
    assert Ts_world_joint.shape[-2:] == (num_joints, 7)
    return Ts_world_joint


def broadcasting_cat(tensors: list[Tensor], dim: int) -> Tensor:
    """Like torch.cat, but broadcasts."""
    assert len(tensors) > 0
    output_dims = max(map(lambda t: len(t.shape), tensors))
    tensors = [
        t.reshape((1,) * (output_dims - len(t.shape)) + t.shape) for t in tensors
    ]
    max_sizes = [max(t.shape[i] for t in tensors) for i in range(output_dims)]
    expanded_tensors = [
        tensor.expand(
            *(
                tensor.shape[i] if i == dim % len(tensor.shape) else max_size
                for i, max_size in enumerate(max_sizes)
            )
        )
        for tensor in tensors
    ]
    return torch.cat(expanded_tensors, dim=dim)


def _normalize_dtype(v: np.ndarray) -> np.ndarray:
    """Normalize datatypes; all arrays should be either int32 or float32."""
    if "int" in str(v.dtype):
        return v.astype(np.int32)
    elif "float" in str(v.dtype):
        return v.astype(np.float32)
    else:
        return v
