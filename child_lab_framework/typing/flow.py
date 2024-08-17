from typing import Protocol

from .stream import Fiber, Input, Output


class Component[Input, Output](Protocol):
    def stream(self) -> Fiber[Input, Output]: ...
