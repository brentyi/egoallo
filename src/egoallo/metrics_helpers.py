from typing import Literal, overload

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from typing_extensions import assert_never

from .transforms import SO3


def compute_foot_skate(
    pred_Ts_world_joint: Float[Tensor, "num_samples time 21 7"],
) -> np.ndarray:
    (num_samples, time) = pred_Ts_world_joint.shape[:2]

    # Drop the person to the floor.
    # This is necessary for the foot skating metric to make sense for floating people...!
    pred_Ts_world_joint = pred_Ts_world_joint.clone()
    pred_Ts_world_joint[..., 6] -= torch.min(pred_Ts_world_joint[..., 6])

    foot_indices = torch.tensor([6, 7, 9, 10], device=pred_Ts_world_joint.device)

    foot_positions = pred_Ts_world_joint[:, :, foot_indices, 4:7]
    foot_positions_diff = foot_positions[:, 1:, :, :2] - foot_positions[:, :-1, :, :2]
    assert foot_positions_diff.shape == (num_samples, time - 1, 4, 2)

    foot_positions_diff_norm = torch.sum(torch.abs(foot_positions_diff), dim=-1)
    assert foot_positions_diff_norm.shape == (num_samples, time - 1, 4)

    # From EgoEgo / kinpoly.
    H_thresh = torch.tensor(
        # To match indices above: (ankle, ankle, toe, toe)
        [0.08, 0.08, 0.04, 0.04],
        device=pred_Ts_world_joint.device,
        dtype=torch.float32,
    )

    foot_positions_diff_norm = torch.sum(torch.abs(foot_positions_diff), dim=-1)
    assert foot_positions_diff_norm.shape == (num_samples, time - 1, 4)

    # Threshold.
    foot_positions_diff_norm = foot_positions_diff_norm * (
        foot_positions[..., 1:, :, 2] < H_thresh
    )
    fs_per_sample = torch.sum(
        torch.sum(
            foot_positions_diff_norm
            * (2 - 2 ** (foot_positions[..., 1:, :, 2] / H_thresh)),
            dim=-1,
        ),
        dim=-1,
    )
    assert fs_per_sample.shape == (num_samples,)

    return fs_per_sample.numpy(force=True)


def compute_foot_contact(
    pred_Ts_world_joint: Float[Tensor, "num_samples time 21 7"],
) -> np.ndarray:
    (num_samples, time) = pred_Ts_world_joint.shape[:2]

    foot_indices = torch.tensor([6, 7, 9, 10], device=pred_Ts_world_joint.device)

    # From EgoEgo / kinpoly.
    H_thresh = torch.tensor(
        # To match indices above: (ankle, ankle, toe, toe)
        [0.08, 0.08, 0.04, 0.04],
        device=pred_Ts_world_joint.device,
        dtype=torch.float32,
    )

    foot_positions = pred_Ts_world_joint[:, :, foot_indices, 4:7]

    any_contact = torch.any(
        torch.any(foot_positions[..., 2] < H_thresh, dim=-1), dim=-1
    ).to(torch.float32)
    assert any_contact.shape == (num_samples,)

    return any_contact.numpy(force=True)


def compute_head_ori(
    label_Ts_world_joint: Float[Tensor, "time 21 7"],
    pred_Ts_world_joint: Float[Tensor, "num_samples time 21 7"],
) -> np.ndarray:
    (num_samples, time) = pred_Ts_world_joint.shape[:2]
    matrix_errors = (
        SO3(pred_Ts_world_joint[:, :, 14, :4]).as_matrix()
        @ SO3(label_Ts_world_joint[:, 14, :4]).inverse().as_matrix()
    ) - torch.eye(3, device=label_Ts_world_joint.device)
    assert matrix_errors.shape == (num_samples, time, 3, 3)

    return torch.mean(
        torch.linalg.norm(matrix_errors.reshape((num_samples, time, 9)), dim=-1),
        dim=-1,
    ).numpy(force=True)


def compute_head_trans(
    label_Ts_world_joint: Float[Tensor, "time 21 7"],
    pred_Ts_world_joint: Float[Tensor, "num_samples time 21 7"],
) -> np.ndarray:
    (num_samples, time) = pred_Ts_world_joint.shape[:2]
    errors = pred_Ts_world_joint[:, :, 14, 4:7] - label_Ts_world_joint[:, 14, 4:7]
    assert errors.shape == (num_samples, time, 3)

    return torch.mean(
        torch.linalg.norm(errors, dim=-1),
        dim=-1,
    ).numpy(force=True)


def compute_mpjpe(
    label_T_world_root: Float[Tensor, "time 7"],
    label_Ts_world_joint: Float[Tensor, "time 21 7"],
    pred_T_world_root: Float[Tensor, "num_samples time 7"],
    pred_Ts_world_joint: Float[Tensor, "num_samples time 21 7"],
    per_frame_procrustes_align: bool,
) -> np.ndarray:
    num_samples, time, _, _ = pred_Ts_world_joint.shape

    # Concatenate the world root to the joints.
    label_Ts_world_joint = torch.cat(
        [label_T_world_root[..., None, :], label_Ts_world_joint], dim=-2
    )
    pred_Ts_world_joint = torch.cat(
        [pred_T_world_root[..., None, :], pred_Ts_world_joint], dim=-2
    )
    del label_T_world_root, pred_T_world_root

    pred_joint_positions = pred_Ts_world_joint[:, :, :, 4:7]
    label_joint_positions = label_Ts_world_joint[None, :, :, 4:7].repeat(
        num_samples, 1, 1, 1
    )

    if per_frame_procrustes_align:
        pred_joint_positions = procrustes_align(
            points_y=pred_joint_positions,
            points_x=label_joint_positions,
            output="aligned_x",
        )

    position_differences = pred_joint_positions - label_joint_positions
    assert position_differences.shape == (num_samples, time, 22, 3)

    # Per-joint position errors, in millimeters.
    pjpe = torch.linalg.norm(position_differences, dim=-1) * 1000.0
    assert pjpe.shape == (num_samples, time, 22)

    # Mean per-joint position errors.
    mpjpe = torch.mean(pjpe.reshape((num_samples, -1)), dim=-1)
    assert mpjpe.shape == (num_samples,)

    return mpjpe.cpu().numpy()


@overload
def procrustes_align(
    points_y: Float[Tensor, "*#batch N 3"],
    points_x: Float[Tensor, "*#batch N 3"],
    output: Literal["transforms"],
    fix_scale: bool = False,
) -> tuple[Tensor, Tensor, Tensor]: ...


@overload
def procrustes_align(
    points_y: Float[Tensor, "*#batch N 3"],
    points_x: Float[Tensor, "*#batch N 3"],
    output: Literal["aligned_x"],
    fix_scale: bool = False,
) -> Tensor: ...


def procrustes_align(
    points_y: Float[Tensor, "*#batch N 3"],
    points_x: Float[Tensor, "*#batch N 3"],
    output: Literal["transforms", "aligned_x"],
    fix_scale: bool = False,
) -> tuple[Tensor, Tensor, Tensor] | Tensor:
    """Similarity transform alignment using the Umeyama method. Adapted from
    SLAHMR: https://github.com/vye16/slahmr/blob/main/slahmr/geometry/pcl.py
    Minimizes:
        mean( || Y - s * (R @ X) + t ||^2 )
    with respect to s, R, and t.
    Returns an (s, R, t) tuple.
    """
    *dims, N, _ = points_y.shape
    device = points_y.device
    N = torch.ones((*dims, 1, 1), device=device) * N

    # subtract mean
    my = points_y.sum(dim=-2) / N[..., 0]  # (*, 3)
    mx = points_x.sum(dim=-2) / N[..., 0]
    y0 = points_y - my[..., None, :]  # (*, N, 3)
    x0 = points_x - mx[..., None, :]

    # correlation
    C = torch.matmul(y0.transpose(-1, -2), x0) / N  # (*, 3, 3)
    U, D, Vh = torch.linalg.svd(C)  # (*, 3, 3), (*, 3), (*, 3, 3)

    S = (
        torch.eye(3, device=device)
        .reshape(*(1,) * (len(dims)), 3, 3)
        .repeat(*dims, 1, 1)
    )
    neg = torch.det(U) * torch.det(Vh.transpose(-1, -2)) < 0
    S = torch.where(
        neg.reshape(*dims, 1, 1),
        S * torch.diag(torch.tensor([1, 1, -1], device=device)),
        S,
    )

    R = torch.matmul(U, torch.matmul(S, Vh))  # (*, 3, 3)

    D = torch.diag_embed(D)  # (*, 3, 3)
    if fix_scale:
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

    assert s.shape == (*dims, 1)
    assert R.shape == (*dims, 3, 3)
    assert t.shape == (*dims, 3)

    if output == "transforms":
        return s, R, t
    elif output == "aligned_x":
        aligned_x = (
            s[..., None, :] * torch.einsum("...ij,...nj->...ni", R, points_x)
            + t[..., None, :]
        )
        assert aligned_x.shape == points_x.shape
        return aligned_x
    else:
        assert_never(output)
