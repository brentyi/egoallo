import abc
from typing import (
    ClassVar,
    Generic,
    Self,
    Tuple,
    Type,
    TypeVar,
    Union,
    final,
    overload,
    override,
)

import numpy as onp
import torch
from torch import Tensor

GroupType = TypeVar("GroupType", bound="MatrixLieGroup")
SEGroupType = TypeVar("SEGroupType", bound="SEBase")


class MatrixLieGroup(abc.ABC):
    """Interface definition for matrix Lie groups."""

    # Class properties.
    # > These will be set in `_utils.register_lie_group()`.

    matrix_dim: ClassVar[int]
    """Dimension of square matrix output from `.as_matrix()`."""

    parameters_dim: ClassVar[int]
    """Dimension of underlying parameters, `.parameters()`."""

    tangent_dim: ClassVar[int]
    """Dimension of tangent space."""

    space_dim: ClassVar[int]
    """Dimension of coordinates that can be transformed."""

    def __init__(
        # Notes:
        # - For the constructor signature to be consistent with subclasses, `parameters`
        #   should be marked as positional-only. But this isn't possible in Python 3.7.
        # - This method is implicitly overriden by the dataclass decorator and
        #   should _not_ be marked abstract.
        self,
        parameters: Tensor,
    ):
        """Construct a group object from its underlying parameters."""
        raise NotImplementedError()

    # Shared implementations.

    @overload
    def __matmul__(self: GroupType, other: GroupType) -> GroupType: ...

    @overload
    def __matmul__(self, other: Tensor) -> Tensor: ...

    def __matmul__(
        self: GroupType, other: Union[GroupType, Tensor]
    ) -> Union[GroupType, Tensor]:
        """Overload for the `@` operator.

        Switches between the group action (`.apply()`) and multiplication
        (`.multiply()`) based on the type of `other`.
        """
        if isinstance(other, (onp.ndarray, Tensor)):
            return self.apply(target=other)
        elif isinstance(other, MatrixLieGroup):
            assert self.space_dim == other.space_dim
            return self.multiply(other=other)
        else:
            assert False, f"Invalid argument type for `@` operator: {type(other)}"

    # Factory.

    @classmethod
    @abc.abstractmethod
    def identity(
        cls: Type[GroupType], device: Union[torch.device, str], dtype: torch.dtype
    ) -> GroupType:
        """Returns identity element.

        Returns:
            Identity element.
        """

    @classmethod
    @abc.abstractmethod
    def from_matrix(cls: Type[GroupType], matrix: Tensor) -> GroupType:
        """Get group member from matrix representation.

        Args:
            matrix: Matrix representaiton.

        Returns:
            Group member.
        """

    # Accessors.

    @abc.abstractmethod
    def as_matrix(self) -> Tensor:
        """Get transformation as a matrix. Homogeneous for SE groups."""

    @abc.abstractmethod
    def parameters(self) -> Tensor:
        """Get underlying representation."""

    # Operations.

    @abc.abstractmethod
    def apply(self, target: Tensor) -> Tensor:
        """Applies group action to a point.

        Args:
            target: Point to transform.

        Returns:
            Transformed point.
        """

    @abc.abstractmethod
    def multiply(self: Self, other: Self) -> Self:
        """Composes this transformation with another.

        Returns:
            self @ other
        """

    @classmethod
    @abc.abstractmethod
    def exp(cls: Type[GroupType], tangent: Tensor) -> GroupType:
        """Computes `expm(wedge(tangent))`.

        Args:
            tangent: Tangent vector to take the exponential of.

        Returns:
            Output.
        """

    @abc.abstractmethod
    def log(self) -> Tensor:
        """Computes `vee(logm(transformation matrix))`.

        Returns:
            Output. Shape should be `(tangent_dim,)`.
        """

    @abc.abstractmethod
    def adjoint(self) -> Tensor:
        """Computes the adjoint, which transforms tangent vectors between tangent
        spaces.

        More precisely, for a transform `GroupType`:
        ```
        GroupType @ exp(omega) = exp(Adj_T @ omega) @ GroupType
        ```

        In robotics, typically used for transforming twists, wrenches, and Jacobians
        across different reference frames.

        Returns:
            Output. Shape should be `(tangent_dim, tangent_dim)`.
        """

    @abc.abstractmethod
    def inverse(self: GroupType) -> GroupType:
        """Computes the inverse of our transform.

        Returns:
            Output.
        """

    @abc.abstractmethod
    def normalize(self: GroupType) -> GroupType:
        """Normalize/projects values and returns.

        Returns:
            GroupType: Normalized group member.
        """

    # @classmethod
    # @abc.abstractmethod
    # def sample_uniform(cls: Type[GroupType], key: Tensor) -> GroupType:
    #     """Draw a uniform sample from the group. Translations (if applicable) are in the
    #     range [-1, 1].
    #
    #     Args:
    #         key: PRNG key, as returned by `jax.random.PRNGKey()`.
    #
    #     Returns:
    #         Sampled group member.
    #     """

    def get_batch_axes(self) -> Tuple[int, ...]:
        """Return any leading batch axes in contained parameters. If an array of shape
        `(100, 4)` is placed in the wxyz field of an SO3 object, for example, this will
        return `(100,)`."""
        return self.parameters().shape[:-1]


class SOBase(MatrixLieGroup):
    """Base class for special orthogonal groups."""


ContainedSOType = TypeVar("ContainedSOType", bound=SOBase)


class SEBase(Generic[ContainedSOType], MatrixLieGroup):
    """Base class for special Euclidean groups.

    Each SE(N) group member contains an SO(N) rotation, as well as an N-dimensional
    translation vector.
    """

    # SE-specific interface.

    @classmethod
    @abc.abstractmethod
    def from_rotation_and_translation(
        cls: Type[SEGroupType],
        rotation: ContainedSOType,
        translation: Tensor,
    ) -> SEGroupType:
        """Construct a rigid transform from a rotation and a translation.

        Args:
            rotation: Rotation term.
            translation: Translation term.

        Returns:
            Constructed transformation.
        """

    @final
    @classmethod
    def from_rotation(cls: Type[SEGroupType], rotation: ContainedSOType) -> SEGroupType:
        return cls.from_rotation_and_translation(
            rotation=rotation,
            translation=rotation.parameters().new_zeros(
                (*rotation.parameters().shape[:-1], cls.space_dim),
                dtype=rotation.parameters().dtype,
            ),
        )

    @abc.abstractmethod
    def rotation(self) -> ContainedSOType:
        """Returns a transform's rotation term."""

    @abc.abstractmethod
    def translation(self) -> Tensor:
        """Returns a transform's translation term."""

    # Overrides.

    @final
    @override
    def apply(self, target: Tensor) -> Tensor:
        return self.rotation() @ target + self.translation()  # type: ignore

    @final
    @override
    def multiply(self: SEGroupType, other: SEGroupType) -> SEGroupType:  # type: ignore
        return type(self).from_rotation_and_translation(
            rotation=self.rotation() @ other.rotation(),
            translation=(self.rotation() @ other.translation()) + self.translation(),
        )

    @final
    @override
    def inverse(self: SEGroupType) -> SEGroupType:
        R_inv = self.rotation().inverse()
        return type(self).from_rotation_and_translation(
            rotation=R_inv,
            translation=-(R_inv @ self.translation()),
        )

    @final
    @override
    def normalize(self: SEGroupType) -> SEGroupType:
        return type(self).from_rotation_and_translation(
            rotation=self.rotation().normalize(),
            translation=self.translation(),
        )
