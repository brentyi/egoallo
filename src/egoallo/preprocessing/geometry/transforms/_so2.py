from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import torch
from typing_extensions import override

from . import _base, hints
from .utils import register_lie_group


@register_lie_group(
    matrix_dim=2,
    parameters_dim=2,
    tangent_dim=1,
    space_dim=2,
)
@dataclasses.dataclass
class SO2(_base.SOBase):
    """Special orthogonal group for 2D rotations.

    Ported to pytorch from `jaxlie.SO2`.

    Internal parameterization is `(cos, sin)`. Tangent parameterization is `(omega,)`.
    """

    # SO2-specific.

    unit_complex: torch.Tensor
    """Internal parameters. `(cos, sin)`."""

    @override
    def __repr__(self) -> str:
        unit_complex = torch.round(self.unit_complex, 5)
        return f"{self.__class__.__name__}(unit_complex={unit_complex})"

    @staticmethod
    def from_radians(theta: hints.Scalar) -> SO2:
        """Construct a rotation object from a scalar angle."""
        theta = torch.as_tensor(theta)
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        return SO2(unit_complex=torch.stack([cos, sin], dim=-1))

    def as_radians(self) -> torch.Tensor:
        """Compute a scalar angle from a rotation object."""
        radians = self.log()[..., 0]
        return radians

    # Factory.

    @staticmethod
    @override
    def Identity(shape: Optional[Tuple] = (), **kwargs) -> SO2:
        id_elem = (
            torch.tensor([1.0, 0.0], **kwargs)
            .reshape(*(1,) * len(shape), 2)
            .repeat(*shape, 1)
        )
        return SO2(unit_complex=id_elem)

    @staticmethod
    @override
    def from_matrix(matrix: torch.Tensor) -> SO2:
        assert matrix.shape[-2:] == (2, 2)
        return SO2(unit_complex=matrix[..., 0])

    # Accessors.

    @override
    def matrix(self) -> torch.Tensor:
        """
        [[cos, -sin], [sin, cos]]
        :returns (*, 2, 2) tensor
        """
        cos, sin = self.unit_complex.unbind(dim=-1)
        return torch.stack([cos, -sin, sin, cos], dim=-1).reshape(*cos.shape, 2, 2)

    @override
    def parameters(self) -> torch.Tensor:
        return self.unit_complex

    # Operations.

    @override
    def act(self, target: torch.Tensor) -> torch.Tensor:
        assert target.shape[-1] == 2
        return torch.einsum("...ij,...j->...i", self.matrix(), target)

    @override
    def mul(self, other: SO2) -> SO2:
        return SO2(
            unit_complex=torch.einsum(
                "...ij,...j->...i", self.matrix(), other.unit_complex
            )
        )

    @staticmethod
    @override
    def exp(tangent: torch.Tensor) -> SO2:
        return SO2(
            unit_complex=torch.stack([torch.cos(tangent), torch.sin(tangent)], dim=-1)
        )

    @override
    def log(self) -> torch.Tensor:
        return torch.atan2(
            self.unit_complex[..., 1, None], self.unit_complex[..., 0, None]
        )

    @override
    def adjoint(self, **kwargs) -> torch.Tensor:
        return torch.eye(1, **kwargs)

    @override
    def inv(self) -> SO2:
        cos, sin = self.unit_complex.unbind(dim=-1)
        return SO2(unit_complex=torch.stack([cos, -sin], dim=-1))

    @override
    def normalize(self) -> SO2:
        return SO2(
            unit_complex=self.unit_complex
            / torch.linalg.norm(self.unit_complex, dim=-1, keepdim=True)
        )
