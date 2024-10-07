"""SMPL-H model, implemented in JAX.

Very little of it is specific to SMPL-H. This could very easily be adapted for other models in SMPL family.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, cast

import jax
import jax_dataclasses as jdc
import jaxlie
import numpy as onp
from einops import einsum
from jax import Array
from jax import numpy as jnp
from jaxtyping import Float, Int


@jdc.pytree_dataclass
class SmplhModel:
    """The SMPL-H human body model."""

    faces: Int[Array, "faces 3"]
    """Vertex indices for mesh faces."""
    J_regressor: Float[Array, "joints+1 verts"]
    """Linear map from vertex to joint positions.
    22 body joints + 2 * 22 hand joints."""
    parent_indices: Int[Array, "joints"]
    """Defines kinematic tree. Index of -1 signifies that a joint is defined
    relative to the root."""
    weights: Float[Array, "verts joints+1"]
    """LBS weights."""
    posedirs: Float[Array, "verts 3 joints*9"]
    """Pose blend shape bases."""
    v_template: Float[Array, "verts 3"]
    """Canonical mesh verts."""
    shapedirs: Float[Array, "verts 3 n_betas"]
    """Shape bases."""

    @staticmethod
    def load(npz_path: Path) -> SmplhModel:
        smplh_params: dict[str, onp.ndarray] = onp.load(npz_path, allow_pickle=True)
        # assert smplh_params["bs_style"].item() == b"lbs"
        # assert smplh_params["bs_type"].item() == b"lrotmin"
        smplh_params = {k: _normalize_dtype(v) for k, v in smplh_params.items()}
        return SmplhModel(
            faces=jnp.array(smplh_params["f"]),
            J_regressor=jnp.array(smplh_params["J_regressor"]),
            parent_indices=jnp.array(smplh_params["kintree_table"][0][1:] - 1),
            weights=jnp.array(smplh_params["weights"]),
            posedirs=jnp.array(smplh_params["posedirs"]),
            v_template=jnp.array(smplh_params["v_template"]),
            shapedirs=jnp.array(smplh_params["shapedirs"]),
        )

    def with_shape(
        self, betas: Float[Array | onp.ndarray, "... n_betas"]
    ) -> SmplhShaped:
        """Compute a new body model, with betas applied. betas vector should
        have shape up to (16,)."""
        num_betas = betas.shape[-1]
        assert num_betas <= 16
        verts_with_shape = self.v_template + einsum(
            self.shapedirs[:, :, :num_betas],
            betas,
            "verts xyz beta, ... beta -> ... verts xyz",
        )
        root_and_joints_pred = einsum(
            self.J_regressor,
            verts_with_shape,
            "joints verts, ... verts xyz -> ... joints xyz",
        )
        root_offset = root_and_joints_pred[..., 0:1, :]
        return SmplhShaped(
            body_model=self,
            verts_zero=verts_with_shape - root_offset,
            joints_zero=root_and_joints_pred[..., 1:, :] - root_offset,
            t_parent_joint=root_and_joints_pred[..., 1:, :]
            - root_and_joints_pred[..., self.parent_indices + 1, :],
        )


@jdc.pytree_dataclass
class SmplhShaped:
    """The SMPL-H body model with a body shape applied."""

    body_model: SmplhModel
    verts_zero: Float[Array, "verts 3"]
    """Vertices of shaped body _relative to the root joint_ at the zero
    configuration."""
    joints_zero: Float[Array, "joints 3"]
    """Joints of shaped body _relative to the root joint_ at the zero
    configuration."""
    t_parent_joint: Float[Array, "joints 3"]
    """Position of each shaped body joint relative to its parent. Does not
    include root."""

    def with_pose_decomposed(
        self,
        T_world_root: Float[Array | onp.ndarray, "7"],
        body_quats: Float[Array | onp.ndarray, "21 4"],
        left_hand_quats: Float[Array | onp.ndarray, "15 4"] | None = None,
        right_hand_quats: Float[Array | onp.ndarray, "15 4"] | None = None,
    ) -> SmplhShapedAndPosed:
        """Pose our SMPL-H body model. Returns a set of joint and vertex outputs."""

        if left_hand_quats is None:
            left_hand_quats = jnp.zeros((15, 4)).at[:, 0].set(1.0)
        if right_hand_quats is None:
            right_hand_quats = jnp.zeros((15, 4)).at[:, 0].set(1.0)
        local_quats = broadcasting_cat(
            cast(list[jax.Array], [body_quats, left_hand_quats, right_hand_quats]),
            axis=0,
        )
        assert local_quats.shape[-2:] == (51, 4)
        return self.with_pose(T_world_root, local_quats)

    def with_pose(
        self,
        T_world_root: Float[Array | onp.ndarray, "... 7"],
        local_quats: Float[Array | onp.ndarray, "... num_joints 4"],
    ) -> SmplhShapedAndPosed:
        """Pose our SMPL-H body model. Returns a set of joint and vertex outputs."""

        # Forward kinematics.
        # assert local_quats.shape == (51, 4), local_quats.shape
        parent_indices = self.body_model.parent_indices
        (num_joints,) = parent_indices.shape[-1:]
        num_active_joints, _ = local_quats.shape[-2:]
        assert local_quats.shape[-1] == 4
        assert num_active_joints <= num_joints
        assert self.t_parent_joint.shape[-2:] == (num_joints, 3)

        # Get relative transforms.
        Ts_parent_child = broadcasting_cat(
            [local_quats, self.t_parent_joint[..., :num_active_joints, :]], axis=-1
        )
        assert Ts_parent_child.shape[-2:] == (num_active_joints, 7)

        # Compute one joint at a time.
        def compute_joint(i: int, Ts_world_joint: Array) -> Array:
            T_world_parent = jnp.where(
                parent_indices[i] == -1,
                T_world_root,
                Ts_world_joint[..., parent_indices[i], :],
            )
            return Ts_world_joint.at[..., i, :].set(
                (
                    jaxlie.SE3(T_world_parent) @ jaxlie.SE3(Ts_parent_child[..., i, :])
                ).wxyz_xyz
            )

        Ts_world_joint = jax.lax.fori_loop(
            lower=0,
            upper=num_joints,
            body_fun=compute_joint,
            init_val=jnp.zeros_like(Ts_parent_child),
        )
        assert Ts_world_joint.shape[-2:] == (num_active_joints, 7)

        return SmplhShapedAndPosed(
            shaped_model=self,
            T_world_root=T_world_root,  # type: ignore
            local_quats=local_quats,  # type: ignore
            Ts_world_joint=Ts_world_joint,
        )

    def get_T_head_cpf(self) -> Float[Array, "7"]:
        """Get the central pupil frame with respect to the head (joint 14). This
        assumes that we're using the SMPL-H model."""

        assert self.verts_zero.shape[-2:] == (6890, 3), "Not using SMPL-H model!"
        right_eye = (
            self.verts_zero[..., 6260, :] + self.verts_zero[..., 6262, :]
        ) / 2.0
        left_eye = (self.verts_zero[..., 2800, :] + self.verts_zero[..., 2802, :]) / 2.0

        # CPF is between the two eyes.
        cpf_pos_wrt_head = (right_eye + left_eye) / 2.0 - self.joints_zero[..., 14, :]

        return broadcasting_cat([jaxlie.SO3.identity().wxyz, cpf_pos_wrt_head], axis=-1)


@jdc.pytree_dataclass
class SmplhShapedAndPosed:
    shaped_model: SmplhShaped
    """Underlying shaped body model."""

    T_world_root: Float[Array, "*#batch 7"]
    """Root coordinate frame."""

    local_quats: Float[Array, "*#batch joints 4"]
    """Local joint orientations."""

    Ts_world_joint: Float[Array, "joints 7"]
    """Absolute transform for each joint. Does not include the root."""

    def with_new_T_world_root(
        self, T_world_root: Float[Array, "*#batch 7"]
    ) -> SmplhShapedAndPosed:
        return SmplhShapedAndPosed(
            shaped_model=self.shaped_model,
            T_world_root=T_world_root,
            local_quats=self.local_quats,
            Ts_world_joint=(
                jaxlie.SE3(T_world_root[..., None, :])
                @ jaxlie.SE3(self.T_world_root[..., None, :]).inverse()
                @ jaxlie.SE3(self.Ts_world_joint)
            ).parameters(),
        )

    def lbs(self) -> SmplhMesh:
        assert (
            self.local_quats.shape[0]
            == self.shaped_model.body_model.parent_indices.shape[0]
        ), "It looks like only a partial set of joint rotations was passed into `with_pose()`. We need all of them for LBS."

        # Linear blend skinning with a pose blend shape.
        verts_with_blend = self.shaped_model.verts_zero + einsum(
            self.shaped_model.body_model.posedirs,
            (jaxlie.SO3(self.local_quats).as_matrix() - jnp.eye(3)).flatten(),
            "verts j joints_times_9, ... joints_times_9 -> ... verts j",
        )
        verts_transformed = einsum(
            broadcasting_cat(
                [
                    # (*, 1, 3, 4)
                    jaxlie.SE3(self.T_world_root).as_matrix()[..., None, :3, :],
                    # (*, 51, 3, 4)
                    jaxlie.SE3(self.Ts_world_joint).as_matrix()[..., :3, :],
                ],
                axis=0,
            ),
            self.shaped_model.body_model.weights,
            jnp.pad(
                verts_with_blend[:, None, :]
                - jnp.concatenate(
                    [
                        jnp.zeros((1, 1, 3)),  # Root joint.
                        self.shaped_model.joints_zero[None, :, :],
                    ],
                    axis=1,
                ),
                ((0, 0), (0, 0), (0, 1)),
                constant_values=1.0,
            ),
            "joints_p1 i j, ... verts joints_p1, ... verts joints_p1 j -> ... verts i",
        )

        return SmplhMesh(
            posed_model=self,
            verts=verts_transformed,
            faces=self.shaped_model.body_model.faces,
        )


@jdc.pytree_dataclass
class SmplhMesh:
    posed_model: SmplhShapedAndPosed

    verts: Float[Array, "verts 3"]
    """Vertices for mesh."""

    faces: Int[Array, "13776 3"]
    """Faces for mesh."""


def broadcasting_cat(arrays: Sequence[jax.Array | onp.ndarray], axis: int) -> jax.Array:
    """Like jnp.concatenate, but broadcasts leading axes."""
    assert len(arrays) > 0
    output_dims = max(map(lambda t: len(t.shape), arrays))
    arrays = [t.reshape((1,) * (output_dims - len(t.shape)) + t.shape) for t in arrays]
    max_sizes = [max(t.shape[i] for t in arrays) for i in range(output_dims)]
    expanded_arrays = [
        jnp.broadcast_to(
            array,
            tuple(
                array.shape[i] if i == axis % len(array.shape) else max_size
                for i, max_size in enumerate(max_sizes)
            ),
        )
        for array in arrays
    ]
    return jnp.concatenate(expanded_arrays, axis=axis)


def _normalize_dtype(v: onp.ndarray) -> onp.ndarray:
    """Normalize datatypes; all arrays should be either int32 or float32."""
    if "int" in str(v.dtype):
        return v.astype(onp.int32)
    elif "float" in str(v.dtype):
        return v.astype(onp.float32)
    else:
        return v
