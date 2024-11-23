from collections.abc import Sequence
from typing import Protocol, TypeVar

Self = TypeVar('Self')


class Comparable(Protocol):
    def __lt__(self: Self, other: Self, /) -> bool: ...
    def __le__(self: Self, other: Self, /) -> bool: ...


class Approximable(Protocol):
    @classmethod
    def approximated(
        cls: type[Self],
        items: Sequence[Self | None],
    ) -> list[Self] | None: ...


class Interpolable(Protocol):
    @classmethod
    def interpolated(
        cls,
        items: Sequence[Self | None],
    ) -> list[Self] | None: ...
