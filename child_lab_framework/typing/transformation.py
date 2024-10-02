from typing import Protocol, TypeVar

from .array import FloatArray1, FloatArray2

Input = TypeVar('Input', contravariant=True)


class Transformation(Protocol):
    @property
    def rotation(self) -> FloatArray2: ...

    @property
    def translation(self) -> FloatArray1: ...
