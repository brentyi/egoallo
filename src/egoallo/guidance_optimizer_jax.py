"""Optimize constraints using Levenberg-Marquardt."""

from __future__ import annotations

import os

from .hand_detection_structs import (
    CorrespondedAriaHandWristPoseDetections,
    CorrespondedHamerDetections,
)

# Need to play nice with PyTorch!
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import dataclasses
import time
from functools import partial
from typing import Callable, Literal, Unpack, assert_never, cast

import jax
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import torch
from jax import numpy as jnp
from jaxtyping import Float, Int
from torch import Tensor

from . import fncsmpl, fncsmpl_jax, network
from .transforms._so3 import SO3


def do_guidance_optimization(
    Ts_world_cpf: Float[Tensor, "time 7"],
    traj: network.EgoDenoiseTraj,
    body_model: fncsmpl.SmplhModel,
    guidance_mode: GuidanceMode,
    phase: Literal["inner", "post"],
    hamer_detections: None | CorrespondedHamerDetections,
    aria_detections: None | CorrespondedAriaHandWristPoseDetections,
    verbose: bool,
) -> tuple[network.EgoDenoiseTraj, dict]:
    """Run an optimizer to apply foot contact constraints."""

    assert traj.hand_rotmats is not None
    guidance_params = JaxGuidanceParams.defaults(guidance_mode, phase)

    start_time = time.time()
    quats, debug_info = _optimize_vmapped(
        body=fncsmpl_jax.SmplhModel(
            faces=cast(jax.Array, body_model.faces.numpy(force=True)),
            J_regressor=cast(jax.Array, body_model.J_regressor.numpy(force=True)),
            parent_indices=cast(jax.Array, onp.array(body_model.parent_indices)),
            weights=cast(jax.Array, body_model.weights.numpy(force=True)),
            posedirs=cast(jax.Array, body_model.posedirs.numpy(force=True)),
            v_template=cast(jax.Array, body_model.v_template.numpy(force=True)),
            shapedirs=cast(jax.Array, body_model.shapedirs.numpy(force=True)),
        ),
        Ts_world_cpf=cast(jax.Array, Ts_world_cpf.numpy(force=True)),
        betas=cast(jax.Array, traj.betas.numpy(force=True)),
        body_rotmats=cast(jax.Array, traj.body_rotmats.numpy(force=True)),
        hand_rotmats=cast(jax.Array, traj.hand_rotmats.numpy(force=True)),
        contacts=cast(jax.Array, traj.contacts.numpy(force=True)),
        guidance_params=guidance_params,
        # The hand detections are a torch tensors in a TensorDataclass form. We
        # use dictionaries to convert to pytrees.
        hamer_detections=None
        if hamer_detections is None
        else hamer_detections.as_nested_dict(numpy=True),
        aria_detections=None
        if aria_detections is None
        else aria_detections.as_nested_dict(numpy=True),
        verbose=verbose,
    )
    rotmats = SO3(
        torch.from_numpy(onp.array(quats))
        .to(traj.body_rotmats.dtype)
        .to(traj.body_rotmats.device)
    ).as_matrix()

    print(f"Constraint optimization finished in {time.time() - start_time}sec")
    return dataclasses.replace(
        traj,
        body_rotmats=rotmats[:, :, :21, :],
        hand_rotmats=rotmats[:, :, 21:, :],
    ), debug_info


class _SmplhBodyPosesVar(
    jaxls.Var[jax.Array],
    default_factory=lambda: jnp.concatenate(
        [jnp.ones((21, 1)), jnp.zeros((21, 3))], axis=-1
    ),
    retract_fn=lambda val, delta: (
        jaxlie.SO3(val) @ jaxlie.SO3.exp(delta.reshape(21, 3))
    ).wxyz,
    tangent_dim=21 * 3,
):
    """Variable containing local joint poses for a SMPL-H human."""


class _SmplhSingleHandPosesVar(
    jaxls.Var[jax.Array],
    default_factory=lambda: jnp.concatenate(
        [jnp.ones((15, 1)), jnp.zeros((15, 3))], axis=-1
    ),
    retract_fn=lambda val, delta: (
        jaxlie.SO3(val) @ jaxlie.SO3.exp(delta.reshape(15, 3))
    ).wxyz,
    tangent_dim=15 * 3,
):
    """Variable containing local joint poses for one hand of a SMPL-H human."""


@jdc.jit
def _optimize_vmapped(
    Ts_world_cpf: jax.Array,
    body: fncsmpl_jax.SmplhModel,
    betas: jax.Array,
    body_rotmats: jax.Array,
    hand_rotmats: jax.Array,
    contacts: jax.Array,
    guidance_params: JaxGuidanceParams,
    hamer_detections: dict | None,
    aria_detections: dict | None,
    verbose: jdc.Static[bool],
) -> tuple[jax.Array, dict]:
    return jax.vmap(
        partial(
            _optimize,
            Ts_world_cpf=Ts_world_cpf,
            body=body,
            guidance_params=guidance_params,
            hamer_detections=hamer_detections,
            aria_detections=aria_detections,
            verbose=verbose,
        )
    )(
        betas=betas,
        body_rotmats=body_rotmats,
        hand_rotmats=hand_rotmats,
        contacts=contacts,
    )


# Modes for guidance.
GuidanceMode = Literal[
    # Foot skating only.
    "no_hands",
    # Only use Aria wrist pose.
    "aria_wrist_only",
    # Use Aria wrist pose + HaMeR 3D estimates.
    "aria_hamer",
    # Use only HaMeR 3D estimates.
    "hamer_wrist",
    # Use HaMeR 3D estimates + reprojection.
    "hamer_reproj2",
]


@jdc.pytree_dataclass
class JaxGuidanceParams:
    prior_quat_weight: float = 1.0
    prior_pos_weight: float = 5.0
    body_quat_vel_smoothness_weight: float = 5.0
    body_quat_smoothness_weight: float = 1.0
    body_quat_delta_smoothness_weight: float = 10.0
    skate_weight: float = 30.0

    # Note: this should be quite high. If the hand quaternions aren't
    # constrained enough the reprojecction loss can get wild.
    hand_quats: jdc.Static[bool] = True
    hand_quat_weight = 5.0

    hand_quat_priors: jdc.Static[bool] = True
    hand_quat_prior_weight = 0.1
    hand_quat_smoothness_weight = 1.0

    hamer_reproj: jdc.Static[bool] = True
    hand_reproj_weight: float = 1.0

    hamer_wrist_pose: jdc.Static[bool] = True
    hamer_abspos_weight: float = 20.0
    hamer_ori_weight: float = 5.0

    aria_wrists: jdc.Static[bool] = True
    aria_wrist_pos_weight: float = 50.0
    aria_wrist_ori_weight: float = 10.0

    # Optimization parameters.
    lambda_initial: float = 0.1
    max_iters: int = 20

    @staticmethod
    def defaults(
        mode: GuidanceMode,
        phase: Literal["inner", "post"],
    ) -> JaxGuidanceParams:
        if mode == "no_hands":
            return {
                "inner": JaxGuidanceParams(
                    hand_quats=False,
                    hand_quat_priors=False,
                    hamer_reproj=False,
                    hamer_wrist_pose=False,
                    aria_wrists=False,
                    max_iters=5,
                ),
                "post": JaxGuidanceParams(
                    hand_quats=False,
                    hand_quat_priors=False,
                    hamer_reproj=False,
                    hamer_wrist_pose=False,
                    aria_wrists=False,
                    max_iters=20,
                ),
            }[phase]
        elif mode == "aria_wrist_only":
            return {
                "inner": JaxGuidanceParams(
                    hand_quats=False,
                    hand_quat_priors=True,
                    hamer_reproj=False,
                    hamer_wrist_pose=False,
                    aria_wrists=True,
                    max_iters=5,
                ),
                "post": JaxGuidanceParams(
                    hand_quats=False,
                    hand_quat_priors=True,
                    hamer_reproj=False,
                    hamer_wrist_pose=False,
                    aria_wrists=True,
                    max_iters=20,
                ),
            }[phase]
        elif mode == "aria_hamer":
            return {
                "inner": JaxGuidanceParams(
                    hand_quats=True,
                    hand_quat_priors=True,
                    hamer_reproj=False,
                    hamer_wrist_pose=False,
                    aria_wrists=True,
                    max_iters=5,
                ),
                "post": JaxGuidanceParams(
                    hand_quats=True,
                    hand_quat_priors=True,
                    hamer_reproj=False,
                    hamer_wrist_pose=False,
                    aria_wrists=True,
                    max_iters=20,
                ),
            }[phase]
        elif mode == "hamer_wrist":
            return {
                "inner": JaxGuidanceParams(
                    hand_quats=True,
                    hand_quat_priors=True,
                    # NOTE: we turn off reprojection during the inner loop optimization.
                    hamer_reproj=False,
                    hamer_wrist_pose=True,
                    aria_wrists=False,
                    max_iters=5,
                ),
                "post": JaxGuidanceParams(
                    hand_quats=True,
                    hand_quat_priors=True,
                    # Turn on reprojection.
                    hamer_reproj=False,
                    hamer_wrist_pose=True,
                    aria_wrists=False,
                    max_iters=20,
                ),
            }[phase]
        elif mode == "hamer_reproj2":
            return {
                "inner": JaxGuidanceParams(
                    hand_quats=True,
                    hand_quat_priors=True,
                    # NOTE: we turn off reprojection during the inner loop optimization.
                    hamer_reproj=False,
                    hamer_wrist_pose=True,
                    aria_wrists=False,
                    max_iters=5,
                ),
                "post": JaxGuidanceParams(
                    hand_quats=True,
                    hand_quat_priors=True,
                    # Turn on reprojection.
                    hamer_reproj=True,
                    hamer_wrist_pose=True,
                    aria_wrists=False,
                    max_iters=20,
                ),
            }[phase]
        else:
            assert_never(mode)


def _optimize(
    Ts_world_cpf: jax.Array,
    body: fncsmpl_jax.SmplhModel,
    betas: jax.Array,
    body_rotmats: jax.Array,
    hand_rotmats: jax.Array,
    contacts: jax.Array,
    guidance_params: JaxGuidanceParams,
    hamer_detections: dict | None,
    aria_detections: dict | None,
    verbose: bool,
) -> tuple[jax.Array, dict]:
    """Apply constraints using Levenberg-Marquardt optimizer. Returns updated
    body_rotmats and hand_rotmats matrices."""
    timesteps = body_rotmats.shape[0]
    assert Ts_world_cpf.shape == (timesteps, 7)
    assert body_rotmats.shape == (timesteps, 21, 3, 3)
    assert hand_rotmats.shape == (timesteps, 30, 3, 3)
    assert contacts.shape == (timesteps, 21)
    assert betas.shape == (timesteps, 16)

    init_quats = jaxlie.SO3.from_matrix(
        # body_rotmats
        jnp.concatenate([body_rotmats, hand_rotmats], axis=1)
    ).wxyz
    assert init_quats.shape == (timesteps, 51, 4)

    # Assume body shape is time-invariant.
    shaped_body = body.with_shape(jnp.mean(betas, axis=0))
    T_head_cpf = shaped_body.get_T_head_cpf()
    T_cpf_head = jaxlie.SE3(T_head_cpf).inverse().parameters()
    assert T_cpf_head.shape == (7,)

    init_posed = shaped_body.with_pose(
        jaxlie.SE3.identity(batch_axes=(timesteps,)).wxyz_xyz, init_quats
    )
    T_world_head = jaxlie.SE3(Ts_world_cpf) @ jaxlie.SE3(T_cpf_head)
    T_root_head = jaxlie.SE3(init_posed.Ts_world_joint[:, 14])
    init_posed = init_posed.with_new_T_world_root(
        (T_world_head @ T_root_head.inverse()).wxyz_xyz
    )
    del T_world_head
    del T_root_head

    foot_joint_indices = jnp.array([6, 7, 9, 10])
    num_foot_joints = foot_joint_indices.shape[0]

    contacts = contacts[..., foot_joint_indices]
    pairwise_contacts = (contacts[:-1, :] + contacts[1:, :]) / 2.0
    assert pairwise_contacts.shape == (timesteps - 1, num_foot_joints)
    del contacts

    # We'll populate a list of factors (cost terms).
    factors = list[jaxls.Factor]()

    def cost_with_args[*CostArgs](
        *args: Unpack[tuple[*CostArgs]],
    ) -> Callable[
        [Callable[[jaxls.VarValues, *CostArgs], jax.Array]],
        Callable[[jaxls.VarValues, *CostArgs], jax.Array],
    ]:
        """Decorator for appending to the factor list."""

        def inner(
            cost_func: Callable[[jaxls.VarValues, *CostArgs], jax.Array],
        ) -> Callable[[jaxls.VarValues, *CostArgs], jax.Array]:
            factors.append(jaxls.Factor.make(cost_func, args))
            return cost_func

        return inner

    def do_forward_kinematics(
        vals: jaxls.VarValues,
        var: _SmplhBodyPosesVar,
        left_hand: _SmplhSingleHandPosesVar | None = None,
        right_hand: _SmplhSingleHandPosesVar | None = None,
        output_frame: Literal["world", "root"] = "world",
    ) -> fncsmpl_jax.SmplhShapedAndPosed:
        """Helper for computing forward kinematics from variables."""
        assert (left_hand is None) == (right_hand is None)
        if left_hand is None and right_hand is None:
            posed = shaped_body.with_pose(
                T_world_root=jaxlie.SE3.identity().wxyz_xyz,
                local_quats=vals[var],
            )
        elif left_hand is not None and right_hand is None:
            posed = shaped_body.with_pose(
                T_world_root=jaxlie.SE3.identity().wxyz_xyz,
                local_quats=jnp.concatenate([vals[var], vals[left_hand]], axis=-2),
            )
        elif left_hand is not None and right_hand is not None:
            posed = shaped_body.with_pose(
                T_world_root=jaxlie.SE3.identity().wxyz_xyz,
                local_quats=jnp.concatenate(
                    [vals[var], vals[left_hand], vals[right_hand]], axis=-2
                ),
            )
        else:
            assert False

        if output_frame == "world":
            T_world_root = (
                # T_world_cpf
                jaxlie.SE3(Ts_world_cpf[var.id, :])
                # T_cpf_head
                @ jaxlie.SE3(T_cpf_head)
                # T_head_root
                @ jaxlie.SE3(posed.Ts_world_joint[14]).inverse()
            )
            return posed.with_new_T_world_root(T_world_root.wxyz_xyz)
        elif output_frame == "root":
            return posed

    # HaMeR pose cost.
    if hamer_detections is not None and guidance_params.hand_quat_priors:
        hamer_left = hamer_detections["detections_left_concat"]
        hamer_right = hamer_detections["detections_right_concat"]

        # HaMeR local quaternion smoothness.
        @(
            cost_with_args(
                _SmplhSingleHandPosesVar(jnp.arange(timesteps * 2 - 2)),
                _SmplhSingleHandPosesVar(jnp.arange(2, timesteps * 2)),
            )
        )
        def hand_smoothness(
            vals: jaxls.VarValues,
            hand_pose: _SmplhSingleHandPosesVar,
            hand_pose_next: _SmplhSingleHandPosesVar,
        ) -> jax.Array:
            return (
                guidance_params.hand_quat_smoothness_weight
                * (
                    jaxlie.SO3(vals[hand_pose]).inverse()
                    @ jaxlie.SO3(vals[hand_pose_next])
                )
                .log()
                .flatten()
            )

        # Hand prior loss.
        @cost_with_args(
            _SmplhSingleHandPosesVar(jnp.arange(timesteps * 2)),
            init_quats[:, 21:51, :].reshape((timesteps * 2, 15, 4)),
        )
        def hand_prior(
            vals: jaxls.VarValues,
            hand_pose: _SmplhSingleHandPosesVar,
            init_hand_quats: jax.Array,
        ) -> jax.Array:
            return (
                guidance_params.hand_quat_prior_weight
                * (jaxlie.SO3(vals[hand_pose]).inverse() @ jaxlie.SO3(init_hand_quats))
                .log()
                .flatten()
            )

    if hamer_detections is not None and guidance_params.hand_quats:
        hamer_left = hamer_detections["detections_left_concat"]
        hamer_right = hamer_detections["detections_right_concat"]

        # HaMeR local pose matching.
        @(
            cost_with_args(
                _SmplhSingleHandPosesVar(hamer_left["indices"] * 2),
                hamer_left["single_hand_quats"],
            )
            if hamer_left is not None
            else lambda x: x
        )
        @(
            cost_with_args(
                _SmplhSingleHandPosesVar(hamer_right["indices"] * 2 + 1),
                hamer_right["single_hand_quats"],
            )
            if hamer_right is not None
            else lambda x: x
        )
        def hamer_local_pose_cost(
            vals: jaxls.VarValues,
            hand_pose: _SmplhSingleHandPosesVar,
            estimated_hand_quats: jax.Array,
        ) -> jax.Array:
            hand_quats = vals[hand_pose]
            assert hand_quats.shape == estimated_hand_quats.shape
            return guidance_params.hand_quat_weight * (
                (jaxlie.SO3(hand_quats).inverse() @ jaxlie.SO3(estimated_hand_quats))
                .log()
                .flatten()
            )

    if hamer_detections is not None and (
        guidance_params.hamer_reproj and guidance_params.hamer_wrist_pose
    ):
        hamer_left = hamer_detections["detections_left_concat"]
        hamer_right = hamer_detections["detections_right_concat"]

        # HaMeR reprojection.
        mano_from_openpose_indices = _get_mano_from_openpose_indices(include_tips=False)

        @(
            cost_with_args(
                _SmplhBodyPosesVar(hamer_left["indices"]),
                _SmplhSingleHandPosesVar(hamer_left["indices"] * 2),
                _SmplhSingleHandPosesVar(hamer_left["indices"] * 2 + 1),
                jnp.full_like(hamer_left["indices"], fill_value=0),
                hamer_left["keypoints_3d"],
                hamer_left["mano_hand_global_orient"],
            )
            if hamer_left is not None
            else lambda x: x
        )
        @(
            cost_with_args(
                _SmplhBodyPosesVar(hamer_right["indices"]),
                _SmplhSingleHandPosesVar(hamer_right["indices"] * 2),
                _SmplhSingleHandPosesVar(hamer_right["indices"] * 2 + 1),
                jnp.full_like(hamer_right["indices"], fill_value=1),
                hamer_right["keypoints_3d"],
                hamer_right["mano_hand_global_orient"],
            )
            if hamer_right is not None
            else lambda x: x
        )
        def hamer_wrist_and_reproj(
            vals: jaxls.VarValues,
            body_pose: _SmplhBodyPosesVar,
            left_hand_pose: _SmplhSingleHandPosesVar,
            right_hand_pose: _SmplhSingleHandPosesVar,
            left0_right1: jax.Array,  # Set to 0 for left, 1 for right.
            keypoints3d_wrt_cam: jax.Array,  # These are in OpenPose order!!
            Rmat_cam_wrist: jax.Array,
        ) -> jax.Array:
            posed = do_forward_kinematics(
                # The right hand comes _after_ the left hand, we can exclude it.
                vals,
                body_pose,
                left_hand_pose,
                right_hand_pose,
                output_frame="root",
            )
            Ts_root_joint = posed.Ts_world_joint  # Sorry for the naming...
            del posed

            # 19 for left wrist, 20 for right wrist.
            wrist_index = 19 + left0_right1
            hand_start_index = 21 + 15 * left0_right1

            assert Ts_root_joint.shape == (51, 7)
            joint_positions_wrt_root = Ts_root_joint[:, 4:7]
            mano_joints_wrt_root = jnp.concatenate(
                [
                    jax.lax.dynamic_slice_in_dim(
                        joint_positions_wrt_root,
                        start_index=wrist_index,
                        slice_size=1,
                        axis=-2,
                    ),
                    jax.lax.dynamic_slice_in_dim(
                        joint_positions_wrt_root,
                        start_index=hand_start_index,
                        slice_size=15,
                        axis=-2,
                    ),
                ],
                axis=0,
            )
            assert mano_joints_wrt_root.shape == (16, 3)
            assert keypoints3d_wrt_cam.shape == (21, 3)  # In OpenPose.

            T_cam_root = (
                # T_cam_cpf (7,)
                jaxlie.SE3(hamer_detections["T_cpf_cam"]).inverse()
                # T_cpf_head (7,)
                @ jaxlie.SE3(T_cpf_head)
                # T_head_root (7,)
                @ jaxlie.SE3(Ts_root_joint[14, :]).inverse()
            )
            assert T_cam_root.parameters().shape == (7,)
            mano_joints_wrt_cam = T_cam_root @ mano_joints_wrt_root
            obs_joints_wrt_cam = keypoints3d_wrt_cam[mano_from_openpose_indices, :]

            mano_uv_wrt_cam = mano_joints_wrt_cam[:, :2] / mano_joints_wrt_cam[:, 2:3]
            obs_uv_wrt_cam = obs_joints_wrt_cam[:, :2] / obs_joints_wrt_cam[:, 2:3]

            T_cam_wrist = jaxlie.SE3.from_rotation_and_translation(
                T_cam_root.rotation() @ jaxlie.SO3(Ts_root_joint[wrist_index, :4]),
                mano_joints_wrt_cam[0, :],
            )
            obs_T_cam_wrist = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3.from_matrix(Rmat_cam_wrist),
                obs_joints_wrt_cam[0, :],
            )

            return jnp.concatenate(
                [
                    (T_cam_wrist.inverse() @ obs_T_cam_wrist).log()
                    * jnp.array(
                        [guidance_params.hamer_abspos_weight] * 3
                        + [guidance_params.hamer_ori_weight] * 3
                    ),
                    guidance_params.hand_reproj_weight
                    * (mano_uv_wrt_cam - obs_uv_wrt_cam).flatten(),
                ]
            )
    elif (
        hamer_detections is not None
        and not guidance_params.hamer_reproj
        and guidance_params.hamer_wrist_pose
    ):
        hamer_left = hamer_detections["detections_left_concat"]
        hamer_right = hamer_detections["detections_right_concat"]

        @(
            cost_with_args(
                _SmplhBodyPosesVar(hamer_left["indices"]),
                jnp.full_like(hamer_left["indices"], fill_value=0),
                hamer_left["keypoints_3d"],
                hamer_left["mano_hand_global_orient"],
            )
            if hamer_left is not None
            else lambda x: x
        )
        @(
            cost_with_args(
                _SmplhBodyPosesVar(hamer_right["indices"]),
                jnp.full_like(hamer_right["indices"], fill_value=1),
                hamer_right["keypoints_3d"],
                hamer_right["mano_hand_global_orient"],
            )
            if hamer_right is not None
            else lambda x: x
        )
        def hamer_wrist_only(
            vals: jaxls.VarValues,
            body_pose: _SmplhBodyPosesVar,
            left0_right1: jax.Array,  # Set to 0 for left, 1 for right.
            keypoints3d_wrt_cam: jax.Array,  # These are in OpenPose order!!
            Rmat_cam_wrist: jax.Array,
        ) -> jax.Array:
            posed = do_forward_kinematics(vals, body_pose, output_frame="root")
            Ts_root_joint = posed.Ts_world_joint  # Sorry for the naming...
            del posed

            # 19 for left wrist, 20 for right wrist.
            wrist_index = 19 + left0_right1

            assert Ts_root_joint.shape == (21, 7)
            wrist_position_wrt_root = Ts_root_joint[wrist_index, 4:7]

            T_cam_root = (
                # T_cam_cpf (7,)
                jaxlie.SE3(hamer_detections["T_cpf_cam"]).inverse()
                # T_cpf_head (7,)
                @ jaxlie.SE3(T_cpf_head)
                # T_head_root (7,)
                @ jaxlie.SE3(Ts_root_joint[14, :]).inverse()
            )
            assert T_cam_root.parameters().shape == (7,)
            wrist_position_wrt_cam = T_cam_root @ wrist_position_wrt_root

            # Assumes OpenPose root is same as Mano root!!
            wrist_pos_wrt_cam = keypoints3d_wrt_cam[0, :]

            T_cam_wrist = jaxlie.SE3.from_rotation_and_translation(
                T_cam_root.rotation() @ jaxlie.SO3(Ts_root_joint[wrist_index, :4]),
                wrist_position_wrt_cam,
            )
            obs_T_cam_wrist = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3.from_matrix(Rmat_cam_wrist),
                wrist_pos_wrt_cam,
            )
            return (T_cam_wrist.inverse() @ obs_T_cam_wrist).log() * jnp.array(
                [guidance_params.hamer_abspos_weight] * 3
                + [guidance_params.hamer_ori_weight] * 3
            )

    # Wrist pose cost.
    if aria_detections is not None and guidance_params.aria_wrists:
        aria_left = aria_detections["detections_left_concat"]
        aria_right = aria_detections["detections_right_concat"]

        @(
            cost_with_args(
                _SmplhBodyPosesVar(aria_left["indices"]),
                aria_left["confidence"],
                aria_left["wrist_position"],
                aria_left["palm_position"],
                aria_left["palm_normal"],
                jnp.full_like(aria_left["indices"], fill_value=0),
            )
            if aria_left is not None
            else lambda x: x
        )
        @(
            cost_with_args(
                _SmplhBodyPosesVar(aria_right["indices"]),
                aria_right["confidence"],
                aria_right["wrist_position"],
                aria_right["palm_position"],
                aria_right["palm_normal"],
                jnp.full_like(aria_right["indices"], fill_value=1),
            )
            if aria_right is not None
            else lambda x: x
        )
        def wrist_pose_cost(
            vals: jaxls.VarValues,
            pose: _SmplhBodyPosesVar,
            confidence: jax.Array,
            wrist_position: jax.Array,
            palm_position: jax.Array,
            palm_normal: jax.Array,
            left0_right1: jax.Array,  # Set to 0 for left, 1 for right.
        ) -> jax.Array:
            assert wrist_position.shape == (3,)
            assert left0_right1.shape == ()
            posed = do_forward_kinematics(vals, pose)

            T_world_wrist = posed.Ts_world_joint[19 + left0_right1]

            pos_cost = (
                # Left wrist is joint 19, right is joint 20.
                T_world_wrist[4:7] - wrist_position
            )

            # Estimate wrist orientation from forward + normal directions.
            palm_forward = palm_position - wrist_position
            palm_forward = palm_forward / jnp.linalg.norm(palm_forward)
            palm_normal = palm_normal / jnp.linalg.norm(palm_normal)
            palm_forward = (  # Flip palm forward if right hand.
                palm_forward * jnp.array([1, -1])[left0_right1]
            )
            palm_forward = (  # Gram-schmidt for forward direction.
                palm_forward - jnp.dot(palm_forward, palm_normal) * palm_normal
            )
            estimatedR_world_wrist = jaxlie.SO3.from_matrix(
                jnp.stack(
                    [
                        palm_forward,
                        -palm_normal,
                        jnp.cross(palm_normal, palm_forward),
                    ],
                    axis=1,
                )
            )
            R_world_wrist = jaxlie.SO3(T_world_wrist[:4])
            ori_cost = (estimatedR_world_wrist.inverse() @ R_world_wrist).log()

            return confidence * jnp.concatenate(
                [
                    guidance_params.aria_wrist_pos_weight * pos_cost,
                    guidance_params.aria_wrist_ori_weight * ori_cost,
                ]
            )

    # Per-frame regularization cost.
    @cost_with_args(
        _SmplhBodyPosesVar(jnp.arange(timesteps)),
    )
    def reg_cost(
        vals: jaxls.VarValues,
        pose: _SmplhBodyPosesVar,
    ) -> jax.Array:
        posed = do_forward_kinematics(vals, pose)

        torso_indices = jnp.array([0, 1, 2, 5, 8])
        return jnp.concatenate(
            [
                guidance_params.prior_quat_weight
                * (
                    jaxlie.SO3(vals[pose]).inverse()
                    @ jaxlie.SO3(init_quats[pose.id, :21, :])
                )
                .log()
                .flatten(),
                # Only include some torso joints.
                guidance_params.prior_pos_weight
                * (
                    posed.Ts_world_joint[torso_indices, 4:7]
                    - init_posed.Ts_world_joint[pose.id, torso_indices, 4:7]
                ).flatten(),
            ]
        )

    @cost_with_args(
        _SmplhBodyPosesVar(jnp.arange(timesteps - 1)),
        _SmplhBodyPosesVar(jnp.arange(1, timesteps)),
    )
    def delta_smoothness_cost(
        vals: jaxls.VarValues,
        current: _SmplhBodyPosesVar,
        next: _SmplhBodyPosesVar,
    ) -> jax.Array:
        curdelt = jaxlie.SO3(vals[current]).inverse() @ jaxlie.SO3(
            init_quats[current.id, :21, :]
        )
        nexdelt = jaxlie.SO3(vals[next]).inverse() @ jaxlie.SO3(
            init_quats[next.id, :21, :]
        )
        return jnp.concatenate(
            [
                guidance_params.body_quat_delta_smoothness_weight
                * (curdelt.inverse() @ nexdelt).log().flatten(),
                guidance_params.body_quat_smoothness_weight
                * (jaxlie.SO3(vals[current]).inverse() @ jaxlie.SO3(vals[next]))
                .log()
                .flatten(),
            ]
        )

    @cost_with_args(
        _SmplhBodyPosesVar(jnp.arange(timesteps - 2)),
        _SmplhBodyPosesVar(jnp.arange(1, timesteps - 1)),
        _SmplhBodyPosesVar(jnp.arange(2, timesteps)),
    )
    def vel_smoothness_cost(
        vals: jaxls.VarValues,
        t0: _SmplhBodyPosesVar,
        t1: _SmplhBodyPosesVar,
        t2: _SmplhBodyPosesVar,
    ) -> jax.Array:
        curdelt = jaxlie.SO3(vals[t0]).inverse() @ jaxlie.SO3(vals[t1])
        nexdelt = jaxlie.SO3(vals[t1]).inverse() @ jaxlie.SO3(vals[t2])
        return (
            guidance_params.body_quat_vel_smoothness_weight
            * (curdelt.inverse() @ nexdelt).log().flatten()
        )

    @cost_with_args(
        _SmplhBodyPosesVar(jnp.arange(timesteps - 1)),
        _SmplhBodyPosesVar(jnp.arange(1, timesteps)),
        pairwise_contacts,
    )
    def skating_cost(
        vals: jaxls.VarValues,
        current: _SmplhBodyPosesVar,
        next: _SmplhBodyPosesVar,
        foot_contacts: jax.Array,
    ) -> jax.Array:
        # Do forward kinematics.
        posed_current = do_forward_kinematics(vals, current)
        posed_next = do_forward_kinematics(vals, next)
        footpos_current = posed_current.Ts_world_joint[foot_joint_indices, 4:7]
        footpos_next = posed_next.Ts_world_joint[foot_joint_indices, 4:7]
        assert footpos_current.shape == footpos_next.shape == (num_foot_joints, 3)
        assert foot_contacts.shape == (num_foot_joints,)

        return (
            guidance_params.skate_weight
            * (foot_contacts[:, None] * (footpos_current - footpos_next)).flatten()
        )

    vars_body_pose = _SmplhBodyPosesVar(jnp.arange(timesteps))
    vars_hand_pose = _SmplhSingleHandPosesVar(jnp.arange(timesteps * 2))
    graph = jaxls.FactorGraph.make(
        factors=factors, variables=[vars_body_pose, vars_hand_pose], use_onp=False
    )
    solutions = graph.solve(
        initial_vals=jaxls.VarValues.make(
            [
                vars_body_pose.with_value(init_quats[:, :21, :]),
                vars_hand_pose.with_value(
                    init_quats[:, 21:51, :].reshape((timesteps * 2, 15, 4))
                ),
            ]
        ),
        linear_solver="conjugate_gradient",
        trust_region=jaxls.TrustRegionConfig(
            lambda_initial=guidance_params.lambda_initial
        ),
        termination=jaxls.TerminationConfig(max_iterations=guidance_params.max_iters),
        verbose=verbose,
    )
    out_body_quats = solutions[_SmplhBodyPosesVar]
    assert out_body_quats.shape == (timesteps, 21, 4)
    out_hand_quats = solutions[_SmplhSingleHandPosesVar].reshape((timesteps, 30, 4))
    assert out_hand_quats.shape == (timesteps, 30, 4)
    return (
        jnp.concatenate([out_body_quats, out_hand_quats], axis=-2),
        {},  # Metadata dict that we use for debugging.
    )


def _get_mano_from_openpose_indices(include_tips: bool) -> Int[onp.ndarray, "21"]:
    # https://github.com/geopavlakos/hamer/blob/272d68f176e0ea8a506f761663dd3dca4a03ced0/hamer/models/mano_wrapper.py#L20
    # fmt: off
    mano_to_openpose = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    # fmt: on
    openpose_from_mano_idx = {
        mano_idx: openpose_idx for openpose_idx, mano_idx in enumerate(mano_to_openpose)
    }
    return onp.array(
        [openpose_from_mano_idx[i] for i in range(21 if include_tips else 16)]
    )
