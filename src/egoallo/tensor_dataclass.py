import dataclasses
from typing import Any, Callable, Self, dataclass_transform

import torch


@dataclass_transform()
class TensorDataclass:
    """A lighter version of nerfstudio's TensorDataclass:
    https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/utils/tensor_dataclass.py
    """

    def __init_subclass__(cls) -> None:
        dataclasses.dataclass(cls)

    def to(self, device: torch.device | str) -> Self:
        """Move the tensors in the dataclass to the given device.

        Args:
            device: The device to move to.

        Returns:
            A new dataclass.
        """
        return self.map(lambda x: x.to(device))

    def as_nested_dict(self, numpy: bool) -> dict[str, Any]:
        """Convert the dataclass to a nested dictionary.

        Recurses into lists, tuples, and dictionaries.
        """

        def _to_dict(obj: Any) -> Any:
            if isinstance(obj, TensorDataclass):
                return {k: _to_dict(v) for k, v in vars(obj).items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(_to_dict(v) for v in obj)
            elif isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, torch.Tensor) and numpy:
                return obj.numpy(force=True)
            else:
                return obj

        return _to_dict(self)

    def map(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Self:
        """Apply a function to all tensors in the dataclass.

        Also recurses into lists, tuples, and dictionaries.

        Args:
            fn: The function to apply to each tensor.

        Returns:
            A new dataclass.
        """

        def _map_impl[MapT](
            fn: Callable[[torch.Tensor], torch.Tensor],
            val: MapT,
        ) -> MapT:
            if isinstance(val, torch.Tensor):
                return fn(val)
            elif isinstance(val, TensorDataclass):
                return type(val)(**_map_impl(fn, vars(val)))
            elif isinstance(val, (list, tuple)):
                return type(val)(_map_impl(fn, v) for v in val)
            elif isinstance(val, dict):
                assert type(val) is dict  # No subclass support.
                return {k: _map_impl(fn, v) for k, v in val.items()}  # type: ignore
            else:
                return val

        return _map_impl(fn, self)
