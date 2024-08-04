from typing import TypeVar
from collections.abc import Generator

Output = TypeVar("Output", covariant=True)
Input = TypeVar("Input", contravariant=True)

type Fiber[Input, Output] = Generator[Output, Input, None]
