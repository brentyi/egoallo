from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import torch
from typing_extensions import override

from . import _base
from ._so3 import SO3
from .utils import get_epsilon, register_lie_group


def _skew(omega: torch.Tensor) -> torch.Tensor:
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
@dataclasses.dataclass
class SE3(_base.SEBase[SO3]):
    """Special Euclidean group for proper rigid transforms in 3D.

    Ported to pytorch from `jaxlie.SE3`.

    Internal parameterization is `(qw, qx, qy, qz, x, y, z)`. Tangent parameterization
    is `(vx, vy, vz, omega_x, omega_y, omega_z)`.
    """

    # SE3-specific.

    wxyz_xyz: torch.Tensor
    """Internal parameters. wxyz quaternion followed by xyz translation."""

    @override
    def __repr__(self) -> str:
        quat = torch.round(self.wxyz_xyz[..., :4], decimals=5)
        trans = torch.round(self.wxyz_xyz[..., 4:], decimals=5)
        return f"{self.__class__.__name__}(wxyz={quat}, xyz={trans})"

    # SE-specific.

    @staticmethod
    @override
    def from_rotation_and_translation(
        rotation: SO3,
        translation: torch.Tensor,
    ) -> "SE3":
        assert translation.shape[-1] == 3
        return SE3(wxyz_xyz=torch.cat([rotation.wxyz, translation], dim=-1))

    @override
    @classmethod
    def from_translation(cls, translation: torch.Tensor) -> "SE3":
        return SE3.from_rotation_and_translation(
            SO3.Identity(
                shape=translation.shape[:-1],
                dtype=translation.dtype,
                device=translation.device,
            ),
            translation,
        )

    @override
    def rotation(self) -> SO3:
        return SO3(wxyz=self.wxyz_xyz[..., :4])

    @override
    def translation(self) -> torch.Tensor:
        return self.wxyz_xyz[..., 4:]

    # Factory.

    @staticmethod
    @override
    def Identity(shape: Optional[Tuple] = (), **kwargs) -> "SE3":
        id_elem = (
            torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], **kwargs)
            .reshape(*(1,) * len(shape), 7)
            .repeat(*shape, 1)
        )
        return SE3(wxyz_xyz=id_elem)

    @staticmethod
    @override
    def from_matrix(matrix: torch.Tensor) -> "SE3":
        assert matrix.shape[-2:] == (4, 4)
        # Currently assumes bottom row is [0, 0, 0, 1].
        return SE3.from_rotation_and_translation(
            rotation=SO3.from_matrix(matrix[..., :3, :3]),
            translation=matrix[..., :3, 3],
        )

    # Accessors.

    @override
    def matrix(self) -> torch.Tensor:
        R = self.rotation().matrix()  # (*, 3, 3)
        t = self.translation().unsqueeze(-1)  # (*, 3, 1)
        dims = R.shape[:-2]
        bottom = (
            torch.tensor([0, 0, 0, 1], dtype=R.dtype, device=R.device)
            .reshape(*(1,) * len(dims), 1, 4)
            .repeat(*dims, 1, 1)
        )
        return torch.cat([torch.cat([R, t], dim=-1), bottom], dim=-2)

    @override
    def parameters(self) -> torch.Tensor:
        return self.wxyz_xyz

    # Operations.

    @staticmethod
    @override
    def exp(tangent: torch.Tensor) -> "SE3":
        """
        :param tangent (*, 6)
        """
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se3.hpp#L761

        # (x, y, z, omega_x, omega_y, omega_z)
        *dims, d = tangent.shape
        assert d == 6

        trans, omega = torch.split(tangent, [3, 3], dim=-1)  # (*, 3), (*, 3)

        rotation = SO3.exp(omega)  # (*, 3)
        theta_squared = torch.square(omega).sum(dim=-1)  # (*)
        use_taylor = theta_squared < get_epsilon(theta_squared.dtype)

        theta_squared_safe = torch.where(
            use_taylor,
            torch.ones_like(theta_squared),  # Any non-zero value should do here.
            theta_squared,
        )
        del theta_squared
        theta_safe = torch.sqrt(theta_squared_safe)

        skew_omega = _skew(omega)  # (*, 3, 3)
        I = (
            torch.eye(3, device=omega.device)
            .reshape(*(1,) * len(dims), 3, 3)
            .expand(*dims, 3, 3)
        )
        f1 = (1.0 - torch.cos(theta_safe)) / (theta_squared_safe)
        f2 = (theta_safe - torch.sin(theta_safe)) / (theta_squared_safe * theta_safe)
        V = torch.where(
            use_taylor[..., None, None],
            rotation.matrix(),
            I
            + f1[..., None, None] * skew_omega
            + f2[..., None, None] * torch.matmul(skew_omega, skew_omega),
        )
        return SE3.from_rotation_and_translation(
            rotation=rotation,
            translation=torch.einsum("...ij,...j->...i", V, trans),
        )

    @override
    def log(self) -> torch.Tensor:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se3.hpp#L223
        omega = self.rotation().log()  # (*, 3)
        theta_squared = torch.square(omega).sum(dim=-1)  # (*)
        use_taylor = theta_squared < get_epsilon(theta_squared.dtype)

        theta_squared_safe = torch.where(
            use_taylor,
            torch.ones_like(theta_squared),  # Any non-zero value should do here.
            theta_squared,
        )
        del theta_squared
        theta_safe = torch.sqrt(theta_squared_safe)
        half_theta_safe = theta_safe / 2.0

        skew_omega = _skew(omega)  # (*, 3, 3)
        skew_omega_sq = torch.matmul(skew_omega)
        I = torch.eye(3, device=omega.device).reshape(*(1,) * len(dims), 3, 3)
        f2 = (
            1.0
            - theta_safe
            * torch.cos(half_theta_safe)
            / (2.0 * torch.sin(half_theta_safe))
        ) / theta_squared_safe

        V_inv = torch.where(
            use_taylor,
            I - 0.5 * skew_omega + skew_omega_sq / 12.0,
            I - 0.5 * skew_omega + f2[..., None, None] * skew_omega_sq,
        )
        return torch.cat(
            [torch.einsum("...ij,...j->...i", V_inv, self.translation()), omega], dim=-1
        )

    @override
    def adjoint(self) -> torch.Tensor:
        R = self.rotation().matrix()
        dims = R.shape[:-2]
        # (*, 6, 6)
        return torch.cat(
            [
                torch.cat([R, torch.matmul(_skew(self.translation()), R)], dim=-1),
                torch.cat([torch.zeros((*dims, 3, 3)), R], dim=-1),
            ],
            dim=-2,
        )
