from typing import NamedTuple, Union

import numpy as np
import torch


Array = torch.Tensor
"""Type alias for `torch.Tensor`."""

Scalar = Union[float, Array]
"""Type alias for `Union[float, Array]`."""


class RollPitchYaw(NamedTuple):
    """Tuple containing roll, pitch, and yaw Euler angles."""

    roll: Scalar
    pitch: Scalar
    yaw: Scalar


__all__ = [
    "Array",
    "Scalar",
    "RollPitchYaw",
]
