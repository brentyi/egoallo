from __future__ import annotations

from typing import Optional, Tuple
import dataclasses

import math
import torch
from typing_extensions import override

from . import _base, hints
from .utils import get_epsilon, register_lie_group


@register_lie_group(
    matrix_dim=3,
    parameters_dim=4,
    tangent_dim=3,
    space_dim=3,
)
@dataclasses.dataclass
class SO3(_base.SOBase):
    """Special orthogonal group for 3D rotations.

    Ported to pytorch from `jaxlie.SO3`.

    Internal parameterization is `(qw, qx, qy, qz)`. Tangent parameterization is
    `(omega_x, omega_y, omega_z)`.
    """

    # SO3-specific.

    wxyz: torch.Tensor
    """Internal parameters. `(w, x, y, z)` quaternion."""

    @override
    def __repr__(self) -> str:
        wxyz = torch.round(self.wxyz, decimals=5)
        return f"{self.__class__.__name__}(wxyz={wxyz})"

    @staticmethod
    def from_x_radians(theta: torch.Tensor) -> SO3:
        """
        Generates a x-axis rotation.
        :param theta (tensor) x rotation
        :returns SO3 object
        """
        zero = torch.zeros_like(theta)
        return SO3.exp(torch.stack([theta, zero, zero], dim=-1))

    @staticmethod
    def from_y_radians(theta: torch.Tensor) -> SO3:
        """
        Generates a y-axis rotation.
        :param theta (tensor) y rotation
        :returns SO3 object
        """
        zero = torch.zeros_like(theta)
        return SO3.exp(torch.stack([zero, theta, zero], dim=-1))

    @staticmethod
    def from_z_radians(theta: torch.Tensor) -> SO3:
        """
        Generates a z-axis rotation.
        :param theta (tensor) z rotation
        :returns SO3 object
        """
        zero = torch.zeros_like(theta)
        return SO3.exp(torch.stack([zero, zero, theta], dim=-1))

    @staticmethod
    def from_rpy_radians(
        roll: torch.Tensor,
        pitch: torch.Tensor,
        yaw: torch.Tensor,
    ) -> SO3:
        """
        Generates a transform from a set of Euler angles. Uses the ZYX convention.
        Args:
            roll: X rotation, in radians. Applied first.
            pitch: Y rotation, in radians. Applied second.
            yaw: Z rotation, in radians. Applied last.
        """
        Rz = SO3.from_z_radians(yaw)
        Ry = SO3.from_y_radians(pitch)
        Rx = SO3.from_x_radians(roll)
        return Rz.mul(Ry.mul(Rx))

    @staticmethod
    def from_quaternion_xyzw(xyzw: torch.Tensor) -> SO3:
        """
        Construct a rotation from an `xyzw` quaternion.
        Note that `wxyz` quaternions can be constructed using the default dataclass
        constructor.
        :param xyzw (*, 4) quat in xyzw convention
        :returns SO3 object
        """
        assert xyzw.shape[-1] == 4
        return SO3(torch.roll(xyzw, shift=1, dims=-1))

    def as_quaternion_xyzw(self) -> torch.Tensor:
        """Grab parameters as xyzw quaternion."""
        return torch.roll(self.wxyz, shift=-1, dims=-1)

    def as_rpy_radians(self) -> hints.RollPitchYaw:
        """
        Computes roll, pitch, and yaw angles. Uses the ZYX convention.
        Returns:
            Named tuple containing Euler angles in radians.
        """
        return hints.RollPitchYaw(
            roll=self.compute_roll_radians(),
            pitch=self.compute_pitch_radians(),
            yaw=self.compute_yaw_radians(),
        )

    def compute_roll_radians(self) -> torch.Tensor:
        """
        Compute roll angle. Uses the ZYX convention.
        :returns angle (*) if wxyz is (*, 4)
        """
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_conversion
        q0, q1, q2, q3 = self.wxyz.unbind(dim=-1)
        return torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 ** 2 + q2 ** 2))

    def compute_pitch_radians(self) -> torch.Tensor:
        """
        Compute pitch angle. Uses the ZYX convention.
        :returns angle (*) if wxyz is (*, 4)
        """
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_conversion
        q0, q1, q2, q3 = self.wxyz.unbind(dim=-1)
        return torch.asin(2 * (q0 * q2 - q3 * q1))

    def compute_yaw_radians(self) -> torch.Tensor:
        """
        Compute yaw angle. Uses the ZYX convention.
        :returns angle (*) if wxyz is (*, 4)
        """
        # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_conversion
        q0, q1, q2, q3 = self.wxyz.unbind(dim=-1)
        return torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))

    # Factory.

    @staticmethod
    @override
    def Identity(shape: Optional[Tuple] = (), **kwargs) -> SO3:
        id_elem = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], **kwargs)
            .reshape(*(1,) * len(shape), 4)
            .repeat(*shape, 1)
        )
        return SO3(wxyz=id_elem)

    @staticmethod
    @override
    def from_matrix(matrix: torch.Tensor) -> SO3:
        assert matrix.shape[-2:] == (3, 3)

        # Modified from:
        # > "Converting a Rotation Matrix to a Quaternion" from Mike Day
        # > https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf

        def case0(m):
            t = 1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2]
            q = torch.stack(
                [
                    m[..., 2, 1] - m[..., 1, 2],
                    t,
                    m[..., 1, 0] + m[..., 0, 1],
                    m[..., 0, 2] + m[..., 2, 0],
                ],
                dim=-1,
            )
            return t, q

        def case1(m):
            t = 1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2]
            q = torch.stack(
                [
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] + m[..., 0, 1],
                    t,
                    m[..., 2, 1] + m[..., 1, 2],
                ],
                dim=-1,
            )
            return t, q

        def case2(m):
            t = 1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]
            q = torch.stack(
                [
                    m[..., 1, 0] - m[..., 0, 1],
                    m[..., 0, 2] + m[..., 2, 0],
                    m[..., 2, 1] + m[..., 1, 2],
                    t,
                ],
                dim=-1,
            )
            return t, q

        def case3(m):
            t = 1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
            q = torch.stack(
                [
                    t,
                    m[..., 2, 1] - m[..., 1, 2],
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] - m[..., 0, 1],
                ],
                dim=-1,
            )
            return t, q

        # Compute four cases, then pick the most precise one.
        # Probably worth revisiting this!
        case0_t, case0_q = case0(matrix)
        case1_t, case1_q = case1(matrix)
        case2_t, case2_q = case2(matrix)
        case3_t, case3_q = case3(matrix)

        cond0 = matrix[..., 2, 2] < 0
        cond1 = matrix[..., 0, 0] > matrix[..., 1, 1]
        cond2 = matrix[..., 0, 0] < -matrix[..., 1, 1]

        t = torch.where(
            cond0,
            torch.where(cond1, case0_t, case1_t),
            torch.where(cond2, case2_t, case3_t),
        ).unsqueeze(-1)
        q = torch.where(
            cond0.unsqueeze(-1),
            torch.where(cond1.unsqueeze(-1), case0_q, case1_q),
            torch.where(cond2.unsqueeze(-1), case2_q, case3_q),
        )

        return SO3(wxyz=q * 0.5 / torch.sqrt(t))

    # Accessors.

    @override
    def matrix(self) -> torch.Tensor:
        norm_sq = torch.square(self.wxyz).sum(dim=-1, keepdim=True)
        qvec = self.wxyz * torch.sqrt(2.0 / norm_sq)  # (*, 4)
        Q = torch.einsum("...i,...j->...ij", qvec, qvec)  # (*, 4, 4)
        return torch.stack(
            [
                1.0 - Q[..., 2, 2] - Q[..., 3, 3],
                Q[..., 1, 2] - Q[..., 3, 0],
                Q[..., 1, 3] + Q[..., 2, 0],
                Q[..., 1, 2] + Q[..., 3, 0],
                1.0 - Q[..., 1, 1] - Q[..., 3, 3],
                Q[..., 2, 3] - Q[..., 1, 0],
                Q[..., 1, 3] - Q[..., 2, 0],
                Q[..., 2, 3] + Q[..., 1, 0],
                1.0 - Q[..., 1, 1] - Q[..., 2, 2],
            ],
            dim=-1,
        ).reshape(*qvec.shape[:-1], 3, 3)

    @override
    def parameters(self) -> torch.Tensor:
        return self.wxyz

    # Operations.

    @override
    def act(self, target: torch.Tensor) -> torch.Tensor:
        assert target.shape[-1] == 3
        # Compute using quaternion muls.
        padded_target = torch.cat([torch.ones_like(target[..., :1]), target], dim=-1)
        out = self.mul(SO3(wxyz=padded_target).mul(self.inv()))
        return out.wxyz[..., 1:]

    @override
    def mul(self, other: SO3) -> SO3:
        w0, x0, y0, z0 = self.wxyz.unbind(dim=-1)
        w1, x1, y1, z1 = other.wxyz.unbind(dim=-1)
        wxyz2 = torch.stack(
            [
                -x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1,
                x0 * w1 + y0 * z1 - z0 * y1 + w0 * x1,
                -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1,
                x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1,
            ],
            dim=-1,
        )

        return SO3(wxyz=wxyz2)

    @staticmethod
    @override
    def exp(tangent: torch.Tensor) -> SO3:
        """
        create SO3 object from axis angle tangent vector
        :param tangent (*, 3)
        """
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/so3.hpp#L583

        assert tangent.shape[-1] == 3

        theta_squared = torch.square(tangent).sum(dim=-1)  # (*)
        theta_pow_4 = theta_squared * theta_squared
        use_taylor = theta_squared < get_epsilon(tangent.dtype)

        safe_theta = torch.sqrt(
            torch.where(
                use_taylor,
                torch.ones_like(theta_squared),  # Any constant value should do here.
                theta_squared,
            )
        )
        safe_half_theta = 0.5 * safe_theta

        real_factor = torch.where(
            use_taylor,
            1.0 - theta_squared / 8.0 + theta_pow_4 / 384.0,
            torch.cos(safe_half_theta),
        )

        imaginary_factor = torch.where(
            use_taylor,
            0.5 - theta_squared / 48.0 + theta_pow_4 / 3840.0,
            torch.sin(safe_half_theta) / safe_theta,
        )

        return SO3(
            wxyz=torch.cat(
                [
                    real_factor[..., None],
                    imaginary_factor[..., None] * tangent,
                ],
                dim=-1,
            )
        )

    @override
    def log(self) -> torch.Tensor:
        """
        log map to tangent space
        :return (*, 3) tangent vector
        """
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/so3.hpp#L247

        w, xyz = torch.split(self.wxyz, [1, 3], dim=-1)  # (*, 1), (*, 3)
        norm_sq = torch.square(xyz).sum(dim=-1, keepdim=True)  # (*, 1)
        use_taylor = norm_sq < get_epsilon(norm_sq.dtype)

        norm_safe = torch.sqrt(
            torch.where(
                use_taylor,
                torch.ones_like(norm_sq),  # Any non-zero value should do here.
                norm_sq,
            )
        )
        w_safe = torch.where(use_taylor, w, torch.ones_like(w))
        atan_n_over_w = torch.atan2(
            torch.where(w < 0, -norm_safe, norm_safe),
            torch.abs(w),
        )
        atan_factor = torch.where(
            use_taylor,
            2.0 / w_safe - 2.0 / 3.0 * norm_sq / w_safe ** 3,
            torch.where(
                torch.abs(w) < get_epsilon(w.dtype),
                torch.where(w > 0, 1.0, -1.0) * math.pi / norm_safe,
                2.0 * atan_n_over_w / norm_safe,
            ),
        )

        return atan_factor * xyz

    @override
    def adjoint(self) -> torch.Tensor:
        return self.matrix()

    @override
    def inv(self) -> SO3:
        # Negate complex terms.
        w, xyz = torch.split(self.wxyz, [1, 3], dim=-1)
        return SO3(wxyz=torch.cat([w, -xyz], dim=-1))

    @override
    def normalize(self) -> SO3:
        return SO3(wxyz=self.wxyz / torch.linalg.norm(self.wxyz, dim=-1, keepdim=True))
