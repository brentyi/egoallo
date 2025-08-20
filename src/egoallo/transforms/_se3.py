from __future__ import annotations

from dataclasses import dataclass
from typing import Union, cast, override

import numpy as np
import torch
from torch import Tensor

from . import _base
from ._so3 import SO3
from .utils import get_epsilon, register_lie_group

def are_se3(matrices: Tensor, tol=1e-6):
    """
    Verifies if a tensor of shape (..., 4, 4) contains matrices that belong to the SE(3) group.

    Args:
        matrices (torch.Tensor): The input tensor of shape (..., 4, 4).
        tol (float): Tolerance for floating-point comparisons.

    Returns:
        torch.Tensor: A boolean tensor of shape (...) indicating which matrices are in SE(3).
    """
    # Check if the matrix has the correct shape
    # Check if the last two dimensions are 4x4
    if matrices.shape[-2:] != (4, 4):
        raise ValueError("Input tensor must have dimensions (..., 4, 4).")

    # Extract the rotation components
    R = matrices[..., :3, :3]
    bottom_row = matrices[..., 3, :]

    # Check if the bottom row is [0, 0, 0, 1] for all matrices
    bottom_row_check = torch.all(torch.isclose(bottom_row, torch.tensor([0.0, 0.0, 0.0, 1.0], device=matrices.device), atol=tol), dim=-1)

    # Check if the 3x3 submatrix R is a valid rotation matrix (R in SO(3))
    # 1. R^T * R = I (R is orthogonal)    
    identity_matrix = torch.eye(3, device=matrices.device)
    r_transpose_r = torch.matmul(R.transpose(-1, -2), R)
    orthogonality_check = torch.all(torch.isclose(r_transpose_r, identity_matrix, atol=tol), dim=(-2, -1))

    # 2. det(R) = 1 (R is a proper rotation)
    determinant_check = torch.isclose(torch.det(R), torch.tensor(1.0, device=matrices.device), atol=tol)

    # Combine all checks
    return bottom_row_check & orthogonality_check & determinant_check

def _skew(omega: Tensor) -> Tensor:
    """
    Returns the skew-symmetric form of a length-3 vector.
    :param omega (*, 3)
    :returns (*, 3, 3)
    """

    wx, wy, wz = omega.unbind(dim=-1)
    o = torch.zeros_like(wx)
    return torch.stack(
        [o, -wz, wy, wz, o, -wx, -wy, wx, o],
        dim=-1,
    ).reshape(*wx.shape, 3, 3)


@register_lie_group(
    matrix_dim=4,
    parameters_dim=7,
    tangent_dim=6,
    space_dim=3,
)
@dataclass(frozen=True)
class SE3(_base.SEBase[SO3]):
    """Special Euclidean group for proper rigid transforms in 3D.

    Internal parameterization is `(qw, qx, qy, qz, x, y, z)`. Tangent parameterization
    is `(vx, vy, vz, omega_x, omega_y, omega_z)`.
    """

    # SE3-specific.

    wxyz_xyz: Tensor
    """Internal parameters. wxyz quaternion followed by xyz translation."""

    @override
    def __repr__(self) -> str:
        quat = np.round(self.wxyz_xyz[..., :4].numpy(force=True), 5)
        trans = np.round(self.wxyz_xyz[..., 4:].numpy(force=True), 5)
        return f"{self.__class__.__name__}(wxyz={quat}, xyz={trans})"

    # SE-specific.

    @classmethod
    @override
    def from_rotation_and_translation(
        cls,
        rotation: SO3,
        translation: Tensor,
    ) -> SE3:
        assert translation.shape[-1] == 3
        return SE3(wxyz_xyz=torch.cat([rotation.wxyz, translation], dim=-1))

    @override
    def rotation(self) -> SO3:
        return SO3(wxyz=self.wxyz_xyz[..., :4])

    @override
    def translation(self) -> Tensor:
        return self.wxyz_xyz[..., 4:]

    # Factory.

    @classmethod
    @override
    def identity(cls, device: Union[torch.device, str], dtype: torch.dtype) -> SE3:
        return SE3(
            wxyz_xyz=torch.tensor(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=device, dtype=dtype
            )
        )

    @classmethod
    @override
    def from_matrix(cls, matrix: Tensor, check=False) -> SE3:
        # check if the matrix is indeed SE(3)
        if check:
            assert torch.all(are_se3(matrix)), "Input matrix is not in SE(3)"
        return SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(matrix[..., :3, :3]),
            translation=matrix[..., :3, 3],
        )

    # Accessors.

    @override
    def as_matrix(self) -> Tensor:
        R = self.rotation().as_matrix()  # (*, 3, 3)
        t = self.translation().unsqueeze(-1)  # (*, 3, 1)
        dims = R.shape[:-2]
        bottom = (
            torch.tensor([0, 0, 0, 1], dtype=R.dtype, device=R.device)
            .reshape(*(1,) * len(dims), 1, 4)
            .repeat(*dims, 1, 1)
        )
        return torch.cat([torch.cat([R, t], dim=-1), bottom], dim=-2)

    @override
    def parameters(self) -> Tensor:
        return self.wxyz_xyz

    # Operations.

    @classmethod
    @override
    def exp(cls, tangent: Tensor) -> SE3:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se3.hpp#L761

        # (x, y, z, omega_x, omega_y, omega_z)
        assert tangent.shape[-1] == 6

        rotation = SO3.exp(tangent[..., 3:])

        theta_squared = torch.square(tangent[..., 3:]).sum(dim=-1)  # (*)
        use_taylor = theta_squared < get_epsilon(theta_squared.dtype)

        theta_squared_safe = cast(
            Tensor,
            torch.where(
                use_taylor,
                1.0,  # Any non-zero value should do here.
                theta_squared,
            ),
        )
        del theta_squared
        theta_safe = torch.sqrt(theta_squared_safe)

        skew_omega = _skew(tangent[..., 3:])
        dtype = skew_omega.dtype
        device = skew_omega.device
        V = torch.where(
            use_taylor[..., None, None],
            rotation.as_matrix(),
            (
                torch.eye(3, device=device, dtype=dtype)
                + ((1.0 - torch.cos(theta_safe)) / (theta_squared_safe))[
                    ..., None, None
                ]
                * skew_omega
                + (
                    (theta_safe - torch.sin(theta_safe))
                    / (theta_squared_safe * theta_safe)
                )[..., None, None]
                * torch.einsum("...ij,...jk->...ik", skew_omega, skew_omega)
            ),
        )

        return SE3.from_rotation_and_translation(
            rotation=rotation,
            translation=torch.einsum("...ij,...j->...i", V, tangent[..., :3]),
        )

    @override
    def log(self) -> Tensor:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se3.hpp#L223
        omega = self.rotation().log()
        theta_squared = torch.square(omega).sum(dim=-1)  # (*)
        use_taylor = theta_squared < get_epsilon(theta_squared.dtype)

        skew_omega = _skew(omega)

        # Shim to avoid NaNs in jnp.where branches, which cause failures for
        # reverse-mode AD.
        theta_squared_safe = torch.where(
            use_taylor,
            1.0,  # Any non-zero value should do here.
            theta_squared,
        )
        del theta_squared
        theta_safe = torch.sqrt(theta_squared_safe)
        half_theta_safe = theta_safe / 2.0

        dtype = omega.dtype
        device = omega.device
        V_inv = torch.where(
            use_taylor[..., None, None],
            torch.eye(3, device=device, dtype=dtype)
            - 0.5 * skew_omega
            + torch.matmul(skew_omega, skew_omega) / 12.0,
            (
                torch.eye(3, device=device, dtype=dtype)
                - 0.5 * skew_omega
                + (
                    1.0
                    - theta_safe
                    * torch.cos(half_theta_safe)
                    / (2.0 * torch.sin(half_theta_safe))
                )[..., None, None]
                / theta_squared_safe[..., None, None]
                * torch.matmul(skew_omega, skew_omega)
            ),
        )
        return torch.cat(
            [torch.einsum("...ij,...j->...i", V_inv, self.translation()), omega], dim=-1
        )

    @override
    def adjoint(self) -> Tensor:
        R = self.rotation().as_matrix()
        dims = R.shape[:-2]
        # (*, 6, 6)
        return torch.cat(
            [
                torch.cat([R, torch.matmul(_skew(self.translation()), R)], dim=-1),
                torch.cat([torch.zeros((*dims, 3, 3)), R], dim=-1),
            ],
            dim=-2,
        )
