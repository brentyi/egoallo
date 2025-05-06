import dataclasses
from typing import Optional, Tuple

import torch
import numpy as onp
from typing_extensions import override

from . import _base, hints
from ._so2 import SO2
from .utils import get_epsilon, register_lie_group


@register_lie_group(
    matrix_dim=3,
    parameters_dim=4,
    tangent_dim=3,
    space_dim=2,
)
@dataclasses.dataclass
class SE2(_base.SEBase[SO2]):
    """Special Euclidean group for proper rigid transforms in 2D.

    Ported to pytorch from `jaxlie.SE2`.

    Internal parameterization is `(cos, sin, x, y)`. Tangent parameterization is `(vx,
    vy, omega)`.
    """

    # SE2-specific.

    unit_complex_xy: torch.Tensor
    """Internal parameters. `(cos, sin, x, y)`."""

    @override
    def __repr__(self) -> str:
        unit_complex = torch.round(self.unit_complex_xy[..., :2], decimals=5)
        xy = torch.round(self.unit_complex_xy[..., 2:], decimals=5)
        return f"{self.__class__.__name__}(unit_complex={unit_complex}, xy={xy})"

    @staticmethod
    def from_xy_theta(x: hints.Scalar, y: hints.Scalar, theta: hints.Scalar) -> "SE2":
        """Construct a transformation from standard 2D pose parameters.

        Note that this is not the same as integrating over a length-3 twist.
        """
        cos = torch.cos(torch.as_tensor(theta))
        sin = torch.sin(torch.as_tensor(theta))
        x, y = torch.as_tensor(x), torch.as_tensor(y)
        return SE2(unit_complex_xy=torch.stack([cos, sin, x, y], dim=-1))

    # SE-specific.

    @staticmethod
    @override
    def from_rotation_and_translation(
        rotation: SO2,
        translation: hints.Array,
    ) -> "SE2":
        assert translation.shape[-1] == 2
        return SE2(
            unit_complex_xy=torch.cat([rotation.unit_complex, translation], dim=-1)
        )

    @override
    @classmethod
    def from_translation(cls, translation: torch.Tensor) -> "SE2":
        return SE2.from_rotation_and_translation(
            SO2.Identity(
                shape=translation.shape[:-1],
                dtype=translation.dtype,
                device=translation.device,
            ),
            translation,
        )

    @override
    def rotation(self) -> SO2:
        return SO2(unit_complex=self.unit_complex_xy[..., :2])

    @override
    def translation(self) -> torch.Tensor:
        return self.unit_complex_xy[..., 2:]

    # Factory.

    @staticmethod
    @override
    def Identity(shape: Optional[Tuple] = (), **kwargs) -> "SE2":
        id_elem = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], **kwargs)
            .reshape(*(1,) * len(shape), 4)
            .repeat(*shape, 1)
        )
        return SE2(unit_complex_xy=id_elem)

    @staticmethod
    @override
    def from_matrix(matrix: hints.Array) -> "SE2":
        assert matrix.shape[-2:] == (3, 3)
        # Currently assumes bottom row is [0, 0, 1].
        return SE2.from_rotation_and_translation(
            rotation=SO2.from_matrix(matrix[..., :2, :2]),
            translation=matrix[..., :2, 2],
        )

    # Accessors.

    @override
    def parameters(self) -> torch.Tensor:
        return self.unit_complex_xy

    @override
    def matrix(self) -> torch.Tensor:
        cos, sin, x, y = self.unit_complex_xy.unbind(dim=-1)
        zero = torch.zeros_like(x)
        one = torch.ones_like(x)
        return torch.stack(
            [cos, -sin, x, sin, cos, y, zero, zero, one], dim=-1
        ).reshape(*cos.shape, 3, 3)

    # Operations.

    @staticmethod
    @override
    def exp(tangent: hints.Array) -> "SE2":
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se2.hpp#L558
        # Also see:
        # > http://ethaneade.com/lie.pdf

        assert tangent.shape[-1] == 3

        theta = tangent[..., 2]

        # transform the translation vector
        use_taylor = torch.abs(theta) < get_epsilon(tangent.dtype)
        safe_theta = torch.where(
            use_taylor,
            torch.ones_like(theta),  # Any non-zero value should do here.
            theta,
        )

        theta_sq = theta ** 2
        sin_over_theta = torch.where(
            use_taylor,
            1.0 - theta_sq / 6.0,
            torch.sin(safe_theta) / safe_theta,
        )
        one_minus_cos_over_theta = torch.where(
            use_taylor,
            0.5 * theta - theta * theta_sq / 24.0,
            (1.0 - torch.cos(safe_theta)) / safe_theta,
        )

        V = torch.stack(
            [
                sin_over_theta,
                -one_minus_cos_over_theta,
                one_minus_cos_over_theta,
                sin_over_theta,
            ],
            dim=-1,
        ).reshape(*theta.shape, 2, 2)

        return SE2.from_rotation_and_translation(
            rotation=SO2.from_radians(theta),
            translation=torch.einsum("...ij,...j->...i", V, tangent[..., :2]),
        )

    @override
    def log(self) -> torch.Tensor:
        # Reference:
        # > https://github.com/strasdat/Sophus/blob/a0fe89a323e20c42d3cecb590937eb7a06b8343a/sophus/se2.hpp#L160
        # Also see:
        # > http://ethaneade.com/lie.pdf

        theta = self.rotation().log()[..., 0]

        cos = torch.cos(theta)
        cos_minus_one = cos - 1.0
        half_theta = theta / 2.0
        use_taylor = torch.abs(cos_minus_one) < get_epsilon(theta.dtype)

        safe_cos_minus_one = torch.where(
            use_taylor,
            torch.ones_like(cos_minus_one),  # Any non-zero value should do here.
            cos_minus_one,
        )

        half_theta_over_tan_half_theta = torch.where(
            use_taylor,
            # Taylor approximation.
            1.0 - theta ** 2 / 12.0,
            # Default.
            -(half_theta * onp.sin(theta)) / safe_cos_minus_one,
        )

        V_inv = torch.stack(
            [
                half_theta_over_tan_half_theta,
                half_theta,
                -half_theta,
                half_theta_over_tan_half_theta,
            ],
            dim=-1,
        ).reshape(*theta.shape, 2, 2)

        tangent = torch.cat(
            [
                torch.einsum("...ij,...j->...i", V_inv, self.translation()),
                theta[..., None],
            ]
        )
        return tangent

    @override
    def adjoint(self, **kwargs) -> torch.Tensor:
        cos, sin, x, y = self.unit_complex_xy.unbind(dim=-1)
        zero = torch.zeros_like(x)
        one = torch.ones_like(x)
        return torch.stack(
            [
                cos,
                -sin,
                y,
                sin,
                cos,
                -x,
                zero,
                zero,
                one,
            ],
            dim=-1,
        ).reshape(*x.shape, 3, 3)
