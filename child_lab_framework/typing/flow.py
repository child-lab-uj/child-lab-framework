from typing import Protocol, TypeVar

from .stream import Fiber


Input = TypeVar('Input', contravariant=True)
Output = TypeVar('Output', covariant=True)

class Component[Input, Output](Protocol):
    def stream(self) -> Fiber[Input, Output]: ...
