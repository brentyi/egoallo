import abc
from typing import ClassVar, Generic, Type, TypeVar, Union, overload, Optional, Tuple

import torch
from typing_extensions import final, override

from . import hints

GroupType = TypeVar("GroupType", bound="MatrixLieGroup")
SEGroupType = TypeVar("SEGroupType", bound="SEBase")


class MatrixLieGroup(abc.ABC):
    """Interface definition for matrix Lie groups."""

    # Class properties.
    # > These will be set in `_utils.register_lie_group()`.

    matrix_dim: ClassVar[int]
    """Dimension of square matrix output from `.matrix()`."""

    parameters_dim: ClassVar[int]
    """Dimension of underlying parameters, `.parameters()`."""

    tangent_dim: ClassVar[int]
    """Dimension of tangent space."""

    space_dim: ClassVar[int]
    """Dimension of coordinates that can be transformed."""

    def __init__(self, parameters: torch.Tensor):
        """
        Construct a group object from its underlying parameters.
        Notes:
        - For the constructor signature to be consistent with subclasses, `parameters`
          should be marked as positional-only. But this isn't possible in Python 3.7.
        - This method is implicitly overriden by the dataclass decorator and
          should _not_ be marked abstract.
        """
        raise NotImplementedError()

    # Shared implementations.

    @overload
    def __mul__(self: GroupType, other: GroupType) -> GroupType:
        ...

    @overload
    def __mul__(self, other: hints.Array) -> torch.Tensor:
        ...

    def __mul__(
        self: GroupType, other: Union[GroupType, hints.Array]
    ) -> Union[GroupType, torch.Tensor]:
        """Overload for the `@` operator.

        Switches between the group action (`.act()`) and multiplication
        (`.mul()`) based on the type of `other`.
        """
        if isinstance(other, hints.Array):
            return self.act(target=other)
        elif isinstance(other, MatrixLieGroup):
            assert self.space_dim == other.space_dim
            return self.mul(other=other)
        else:
            assert False, f"Invalid argument type for `@` operator: {type(other)}"

    # Factory.

    @classmethod
    @abc.abstractmethod
    def Identity(
        cls: Type[GroupType], shape: Optional[Tuple] = (), **kwargs
    ) -> GroupType:
        """Returns identity element.

        Returns:
            Identity element.
        """

    @classmethod
    @abc.abstractmethod
    def from_matrix(cls: Type[GroupType], matrix: hints.Array) -> GroupType:
        """Get group member from matrix representation.

        Args:
            matrix: Matrix representaiton.

        Returns:
            Group member.
        """

    # Accessors.

    @abc.abstractmethod
    def matrix(self) -> torch.Tensor:
        """Get transformation as a matrix. Homogeneous for SE groups."""

    @abc.abstractmethod
    def parameters(self) -> torch.Tensor:
        """Get underlying representation."""

    @property
    def data(self) -> torch.Tensor:
        return self.parameters()

    def __getitem__(self, index):
        return self.__class__(self.data[index])

    @property
    def shape(self):
        return self.data.shape

    # Operations.

    @abc.abstractmethod
    def act(self, target: hints.Array) -> torch.Tensor:
        """Applies group action to a point.

        Args:
            target: Point to transform.

        Returns:
            Transformed point.
        """

    @abc.abstractmethod
    def mul(self: GroupType, other: GroupType) -> GroupType:
        """Composes this transformation with another.

        Returns:
            self @ other
        """

    @classmethod
    @abc.abstractmethod
    def exp(cls: Type[GroupType], tangent: hints.Array) -> GroupType:
        """Computes `expm(wedge(tangent))`.

        Args:
            tangent: Tangent vector to take the exponential of.

        Returns:
            Output.
        """

    @abc.abstractmethod
    def log(self) -> torch.Tensor:
        """Computes `vee(logm(transformation matrix))`.

        Returns:
            Output. Shape should be `(tangent_dim,)`.
        """

    @abc.abstractmethod
    def adjoint(self, **kwargs) -> torch.Tensor:
        """Computes the adjoint, which transforms tangent vectors between tangent
        spaces.

        More precisely, for a transform `GroupType`:
        ```
        GroupType @ exp(omega) = exp(Adj_T @ omega) @ GroupType
        ```
        used for e.g. transforming twists, wrenches, and Jacobians
        across different reference frames.

        Returns:
            Output. Shape should be `(tangent_dim, tangent_dim)`.
        """

    @abc.abstractmethod
    def inv(self: GroupType) -> GroupType:
        """Computes the inv of our transform.

        Returns:
            Output.
        """

    @abc.abstractmethod
    def normalize(self: GroupType) -> GroupType:
        """Normalize/projects values and returns.

        Returns:
            GroupType: Normalized group member.
        """


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
        translation: hints.Array,
    ) -> SEGroupType:
        """Construct a rigid transform from a rotation and a translation.

        Args:
            rotation: Rotation term.
            translation: translation term.

        Returns:
            Constructed transformation.
        """

    @final
    @classmethod
    def from_rotation(cls: Type[SEGroupType], rotation: ContainedSOType) -> SEGroupType:
        data = rotation.parameters()
        return cls.from_rotation_and_translation(
            rotation=rotation,
            translation=torch.zeros(
                *data.shape[:-1], cls.space_dim, dtype=data.dtype, device=data.devce
            ),
        )

    @classmethod
    @abc.abstractmethod
    def from_translation(
        cls: Type[SEGroupType], translation: torch.Tensor
    ) -> SEGroupType:
        """Construct a transform from a translation term."""

    @abc.abstractmethod
    def rotation(self) -> ContainedSOType:
        """Returns a transform's rotation term."""

    @abc.abstractmethod
    def translation(self) -> torch.Tensor:
        """Returns a transform's translation term."""

    # Overrides.

    @final
    @override
    def act(self, target: hints.Array) -> torch.Tensor:
        """
        apply transform to point
        """
        d = self.space_dim
        if target.shape[-1] == d:
            return self.rotation().act(target) + self.translation()  # type: ignore

        # homogeneous point
        assert target.shape[-1] == d + 1
        X, W = torch.split(target, [d, 1], dim=-1)  # (*, d), (*, 1)
        Xp = self.rotation().act(X) + W * self.translation()
        return torch.cat([Xp, W], dim=-1)

    @final
    @override
    def mul(self: SEGroupType, other: SEGroupType) -> SEGroupType:
        return type(self).from_rotation_and_translation(
            rotation=self.rotation().mul(other.rotation()),
            translation=self.rotation().act(other.translation()) + self.translation(),
        )

    @final
    @override
    def inv(self: SEGroupType) -> SEGroupType:
        R_inv = self.rotation().inv()
        return type(self).from_rotation_and_translation(
            rotation=R_inv,
            translation=-R_inv.act(self.translation()),
        )

    @final
    @override
    def normalize(self: SEGroupType) -> SEGroupType:
        return type(self).from_rotation_and_translation(
            rotation=self.rotation().normalize(),
            translation=self.translation(),
        )
