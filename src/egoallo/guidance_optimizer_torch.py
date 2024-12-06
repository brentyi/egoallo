"""Optimize constraints using Levenberg-Marquardt."""

from __future__ import annotations

import os

from torch.optim import LBFGS

from egoallo import fncsmpl_extensions
from egoallo.transforms._se3 import SE3

from .guidance_optimizer_jax import GuidanceMode, JaxGuidanceParams
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

import jaxls
import numpy as onp
import torch
import torch.optim
from jaxtyping import Float, Int
from torch import Tensor

from . import fncsmpl, fncsmpl_jax, network
from .guidance_optimizer_jax import _get_mano_from_openpose_indices
from .transforms._so3 import SO3


def do_guidance_optimization_torch(
    Ts_world_cpf: Float[Tensor, "time 7"],
    traj: network.EgoDenoiseTraj,
    body_model: fncsmpl.SmplhModel,
    guidance_mode: GuidanceMode,
    phase: Literal["inner", "post"],
    optimizer: Literal["lbfgs", "adam"],
    hamer_detections: None | CorrespondedHamerDetections,
    aria_detections: None | CorrespondedAriaHandWristPoseDetections,
) -> tuple[network.EgoDenoiseTraj, dict]:
    """Run an optimizer to apply foot contact constraints."""

    assert hamer_detections is not None
    assert aria_detections is not None

    guidance_params = JaxGuidanceParams.defaults(guidance_mode, phase)
    assert len(Ts_world_cpf.shape) == 2

    # if optimizer == "lbfgs":
    device = Ts_world_cpf.device
    timesteps = Ts_world_cpf.shape[0]
    batch = traj.betas.shape[0]
    assert traj.body_rotmats.shape == (batch, timesteps, 21, 3, 3)
    assert traj.hand_rotmats is not None
    assert traj.hand_rotmats.shape == (batch, timesteps, 30, 3, 3)

    body_deltas = torch.zeros(
        (batch, timesteps, 21, 3), requires_grad=True, device=device
    )
    hand_deltas = torch.zeros(
        (batch, timesteps, 30, 3), requires_grad=True, device=device
    )
    initial_body_quats = SO3.from_matrix(traj.body_rotmats).wxyz
    assert traj.hand_rotmats is not None
    initial_hand_quats = SO3.from_matrix(traj.hand_rotmats).wxyz

    assert traj.betas.shape == (batch, timesteps, 16)
    shaped = body_model.with_shape(torch.mean(traj.betas, dim=1))
    T_head_cpf = fncsmpl_extensions.get_T_head_cpf(shaped)
    T_cpf_head = SE3(T_head_cpf).inverse().parameters()
    assert T_cpf_head.shape == (batch, 7)

    initial_posed_wrt_root = shaped.with_pose(
        T_world_root=SE3.identity(device=device, dtype=torch.float32).wxyz_xyz,
        local_quats=torch.cat([initial_body_quats, initial_hand_quats], dim=-2),
    )
    T_world_root = (
        # T_world_cpf (time, 7) => (1, time, 7)
        SE3(Ts_world_cpf[None, :, :])
        # T_cpf_head (batch, 7) => (batch, 1, 7)
        @ SE3(T_cpf_head[:, None, :])
        # T_head_root (batch, time, joint, 7) => (batch, time, 7)
        @ SE3(initial_posed_wrt_root.Ts_world_joint[:, :, 14, :]).inverse()
    )
    initial_posed_wrt_world = initial_posed_wrt_root.with_new_T_world_root(
        T_world_root.wxyz_xyz
    )

    def compute_loss() -> Tensor:
        # Apply the deltas.
        body_quats = (SO3(initial_body_quats) @ SO3.exp(body_deltas)).wxyz
        hand_quats = (SO3(initial_hand_quats) @ SO3.exp(hand_deltas)).wxyz
        assert body_quats.shape == (batch, timesteps, 21, 4)
        assert hand_quats.shape == (batch, timesteps, 30, 4)

        # Get posed human.
        posed_wrt_root = shaped.with_pose(
            T_world_root=SE3.identity(device=device, dtype=torch.float32).wxyz_xyz,
            local_quats=torch.cat([body_quats, hand_quats], dim=-2),
        )
        T_world_root = (
            # T_world_cpf (time, 7) => (1, time, 7)
            SE3(Ts_world_cpf[None, :, :])
            # T_cpf_head (batch, 7) => (batch, 1, 7)
            @ SE3(T_cpf_head[:, None, :])
            # T_head_root (batch, time, joint, 7) => (batch, time, 7)
            @ SE3(posed_wrt_root.Ts_world_joint[:, :, 14, :]).inverse()
        )
        assert T_world_root.wxyz_xyz.shape == (batch, timesteps, 7)
        posed_wrt_world = posed_wrt_root.with_new_T_world_root(
            T_world_root=T_world_root.wxyz_xyz
        )

        loss_terms = dict[str, Tensor]()

        loss_terms["hand_smoothness"] = (
            guidance_params.hand_quat_smoothness_weight
            * (
                SO3(hand_quats[:, 1:, :, :]).inverse() @ SO3(hand_quats[:, :-1, :, :])
            ).log()
        )
        loss_terms["hand_prior"] = (
            guidance_params.hand_quat_prior_weight
            * (SO3(hand_quats) @ SO3(initial_hand_quats).inverse()).log()
        )

        left_detections = hamer_detections.detections_left_concat
        right_detections = hamer_detections.detections_right_concat
        assert left_detections is not None
        assert right_detections is not None
        loss_terms["hamer_local_pose_cost_left"] = (
            guidance_params.hand_quat_weight
            * (
                SO3(hand_quats[:, left_detections.indices, 0:15, :]).inverse()
                @ SO3(left_detections.single_hand_quats[None, :, :, :])
            ).log()
        )
        loss_terms["hamer_local_pose_cost_right"] = (
            guidance_params.hand_quat_weight
            * (
                SO3(hand_quats[:, right_detections.indices, 15:30, :]).inverse()
                @ SO3(right_detections.single_hand_quats[None, :, :, :])
            ).log()
        )

        def get_wrist_and_reproj_loss(left0_right1: Literal[0, 1]) -> dict[str, Tensor]:
            wrist_index = 19 + left0_right1
            hand_start_index = 21 + 15 * left0_right1

            assert hamer_detections.detections_left_concat is not None
            assert hamer_detections.detections_right_concat is not None

            detections = (
                hamer_detections.detections_left_concat
                if left0_right1 == 0
                else hamer_detections.detections_right_concat
            )
            detection_count = len(detections.indices)
            Ts_root_joint = posed_wrt_root.Ts_world_joint[:, detections.indices, :, :]

            mano_joints_wrt_root = torch.cat(
                [
                    Ts_root_joint[:, :, wrist_index : wrist_index + 1, 4:7],
                    Ts_root_joint[:, :, hand_start_index : hand_start_index + 15, 4:7],
                ],
                dim=2,
            )
            assert mano_joints_wrt_root.shape == (batch, detection_count, 16, 3)

            T_cam_root = (
                # T_cam_cpf (7,) => (1, 1, 7)
                SE3(hamer_detections.T_cpf_cam[None, None, :]).inverse()
                # T_cpf_head (batch, 7) => (batch, 1, 7)
                @ SE3(T_cpf_head[:, None, :])
                # T_head_root (batch, timesteps, joints, 7) => (batch, timesteps, 7)
                @ SE3(Ts_root_joint[:, :, 14, :]).inverse()
            )
            assert T_cam_root.wxyz_xyz.shape == (batch, detection_count, 7)
            mano_joints_wrt_cam = (
                SE3(T_cam_root.wxyz_xyz[:, :, None, :]) @ mano_joints_wrt_root
            )
            assert mano_joints_wrt_cam.shape == (batch, detection_count, 16, 3)

            mano_from_openpose_indices = _get_mano_from_openpose_indices(
                include_tips=False
            )
            obs_joints_wrt_cam = detections.keypoints_3d[
                :, mano_from_openpose_indices, :
            ]
            assert obs_joints_wrt_cam.shape == (detection_count, 16, 3)

            T_cam_wrist = SE3.from_rotation_and_translation(
                T_cam_root.rotation() @ SO3(Ts_root_joint[:, :, wrist_index, :4]),
                mano_joints_wrt_cam[:, :, 0, :],
            )
            Rmat_cam_wrist = detections.mano_hand_global_orient
            assert Rmat_cam_wrist.shape == (len(detections.indices), 3, 3)
            obs_T_cam_wrist = SE3.from_rotation_and_translation(
                SO3.from_matrix(Rmat_cam_wrist),
                obs_joints_wrt_cam[:, 0, :],
            )
            out = {}
            if guidance_params.hamer_wrist_pose:
                out[f"wrist{left0_right1}_abspo"] = (
                    T_cam_wrist.inverse() @ obs_T_cam_wrist
                ).log() * torch.tensor(
                    [guidance_params.hamer_abspos_weight] * 3
                    + [guidance_params.hamer_ori_weight] * 3,
                    device=device,
                    dtype=torch.float32,
                )

            if guidance_params.hamer_reproj:
                mano_uv_wrt_cam = (
                    mano_joints_wrt_cam[..., :2] / mano_joints_wrt_cam[..., 2:3]
                )
                obs_uv_wrt_cam = (
                    obs_joints_wrt_cam[..., :2] / obs_joints_wrt_cam[..., 2:3]
                )
                out[f"reproj{left0_right1}_abspo"] = (
                    mano_uv_wrt_cam[:, :, :, :] - obs_uv_wrt_cam[None, :, :, :]
                ).flatten() * guidance_params.hand_reproj_weight
            return out

        # HaMeR wrist and reprojection loss.
        loss_terms.update(get_wrist_and_reproj_loss(0))
        loss_terms.update(get_wrist_and_reproj_loss(1))

        # Aria wrist pose cost.
        def get_aria_wrist_pose_loss(left0_right1: Literal[0, 1]) -> dict[str, Tensor]:
            out = {}

            detections = (
                aria_detections.detections_left_concat
                if left0_right1 == 0
                else aria_detections.detections_right_concat
            )
            assert detections is not None
            num_detected = len(detections.indices)

            T_world_wrist = posed_wrt_world.Ts_world_joint[
                :, detections.indices, 19 + left0_right1, :
            ]
            assert T_world_wrist.shape == (batch, num_detected, 7)
            assert detections.wrist_position.shape == (num_detected, 3)

            pos_cost = (
                # Left wrist is joint 19, right is joint 20.
                T_world_wrist[:, :, 4:7] - detections.wrist_position[None, :, :]
            )

            palm_forward = detections.palm_position - detections.wrist_position
            assert palm_forward.shape == (num_detected, 3)
            palm_forward = palm_forward / torch.linalg.norm(
                palm_forward, dim=-1, keepdim=True
            )
            palm_normal = detections.palm_normal / torch.linalg.norm(
                detections.palm_normal, dim=-1, keepdim=True
            )
            assert palm_normal.shape == (num_detected, 3)
            if left0_right1 == 1:
                palm_forward = -palm_forward
            # palm_forward = (  # Flip palm forward if right hand.
            #     palm_forward
            #     * torch.tensor([1, -1], device=device, dtype=torch.float32)[
            #         left0_right1
            #     ]
            # )
            palm_forward = (  # Gram-schmidt for forward direction.
                palm_forward
                - torch.sum(palm_forward * palm_normal, dim=-1, keepdim=True)
                * palm_normal
            )
            assert palm_forward.shape == (num_detected, 3)
            estimatedR_world_wrist = SO3.from_matrix(
                torch.stack(
                    [
                        palm_forward,
                        -palm_normal,
                        torch.cross(palm_normal, palm_forward),
                    ],
                    dim=2,
                )
            )
            assert estimatedR_world_wrist.wxyz.shape == (len(detections.indices), 4)
            ori_cost = (
                SO3(estimatedR_world_wrist.wxyz[None, :, :]).inverse()
                @ SO3(T_world_wrist[:, :, :4])
            ).log()
            assert ori_cost.shape == (batch, num_detected, 3)
            assert pos_cost.shape == (batch, num_detected, 3)
            assert detections.confidence.shape == (num_detected,)

            return {
                f"aria_wrist_pos_cost{left0_right1}": (
                    detections.confidence[None, :, None] * pos_cost
                ).flatten()
                * guidance_params.aria_wrist_pos_weight,
                f"aria_wrist_ori_cost{left0_right1}": (
                    detections.confidence[None, :, None] * ori_cost
                ).flatten()
                * guidance_params.aria_wrist_ori_weight,
            }

        if guidance_params.aria_wrists:
            loss_terms.update(get_aria_wrist_pose_loss(0))
            loss_terms.update(get_aria_wrist_pose_loss(1))

        torso_indices = torch.tensor([0, 1, 2, 5, 8], device=device, dtype=torch.int64)
        loss_terms["reg_cost"] = torch.cat(
            [
                guidance_params.prior_quat_weight
                * (SO3(body_quats).inverse() @ SO3(initial_body_quats)).log().flatten(),
                # Only penalize the torso.
                guidance_params.prior_pos_weight
                * (
                    posed_wrt_world.Ts_world_joint[:, :, torso_indices, 4:7]
                    - initial_posed_wrt_world.Ts_world_joint[:, :, torso_indices, 4:7]
                ).flatten(),
            ]
        )

        body_quats_current = body_quats[:, :-1, :, :]
        body_quats_next = body_quats[:, 1:, :, :]
        curdelt = SO3(body_quats_current).inverse() @ SO3(
            initial_body_quats[:, :-1, :, :]
        )
        nexdelt = SO3(body_quats_next).inverse() @ SO3(initial_body_quats[:, 1:, :, :])
        loss_terms["delta_smoothness_cost"] = torch.cat(
            [
                guidance_params.body_quat_delta_smoothness_weight
                * (curdelt.inverse() @ nexdelt).log().flatten(),
                guidance_params.body_quat_smoothness_weight
                * (SO3(body_quats_current).inverse() @ SO3(body_quats_next))
                .log()
                .flatten(),
            ]
        )

        t0 = body_quats[:, :-2, :, :]
        t1 = body_quats[:, 1:-1, :, :]
        t2 = body_quats[:, 2:, :, :]
        curdelt = SO3(t0).inverse() @ SO3(t1)
        nexdelt = SO3(t1).inverse() @ SO3(t2)
        loss_terms["vel_smoothness_cost"] = torch.cat(
            [
                guidance_params.body_quat_vel_smoothness_weight
                * (curdelt.inverse() @ nexdelt).log().flatten(),
            ]
        )

        foot_joint_indices = torch.tensor(
            [6, 7, 9, 10], device=device, dtype=torch.int64
        )
        num_foot_joints = foot_joint_indices.shape[0]

        contacts = traj.contacts[..., foot_joint_indices]
        assert contacts.shape == (batch, timesteps, num_foot_joints)
        pairwise_contacts = (contacts[:, :-1, :] + contacts[:, 1:, :]) / 2.0
        assert pairwise_contacts.shape == (batch, timesteps - 1, num_foot_joints)

        footpos_current = posed_wrt_world.Ts_world_joint[
            :, :-1, foot_joint_indices, 4:7
        ]
        footpos_next = posed_wrt_world.Ts_world_joint[:, 1:, foot_joint_indices, 4:7]
        assert footpos_current.shape == (batch, timesteps - 1, num_foot_joints, 3)
        assert footpos_next.shape == (batch, timesteps - 1, num_foot_joints, 3)

        loss_terms["skating_cost"] = (
            guidance_params.skate_weight
            * pairwise_contacts[:, :, :, None]
            * (footpos_current - footpos_next)[:, None, :, :]
        )

        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        for key, term in loss_terms.items():
            sse = torch.sum(term**2)
            total_loss += sse

        return total_loss

        # loss_terms.update(compute_wr

    start_time = time.time()
    if optimizer == "lbfgs":
        opt = LBFGS(
            [body_deltas, hand_deltas],
            lr=1.0,
            max_iter=100,
            line_search_fn="strong_wolfe",
            history_size=10,
        )
        for i in range(100):

            last_loss = None

            def closure():
                opt.zero_grad()
                loss = compute_loss()
                loss.backward()

                nonlocal last_loss
                last_loss = loss.item()

                return loss

            opt.step(closure)
            time_elapsed = time.time() - start_time
            print(f"{i=}, {time_elapsed=:.2f}, loss={last_loss:.6f}")

    elif optimizer == "adam":
        opt = torch.optim.Adam([body_deltas, hand_deltas], lr=1e-3)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.99)

        for i in range(1000):  # Increased iterations for Adam
            opt.zero_grad()
            loss = compute_loss()
            loss.backward()
            opt.step()
            # scheduler.step()

            if i % 10 == 0:  # Print every 10 iterations
                time_elapsed = time.time() - start_time
                print(f"{i=}, {time_elapsed=:.2f}, loss={loss.item():.6f}")

            if i > 100 and loss.item() < 1e-6:  # Early stopping conditionca
                print("Converged. Stopping early.")
                break

    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    final_loss = compute_loss().item()
    print(f"Final loss: {final_loss:.6f}")

    # # Apply the optimized deltas to get the final pose
    # body_quats = (SO3(initial_body_quats) @ SO3.exp(body_deltas)).wxyz
    # hand_quats = (SO3(initial_hand_quats) @ SO3.exp(hand_deltas)).wxyz
    #
    # # Update the trajectory with the optimized pose
    # traj.body_rotmats = SO3(body_quats).matrix
    # traj.hand_rotmats = SO3(hand_quats).matrix
    #
    return traj, {"final_loss": final_loss}
