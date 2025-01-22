from collections.abc import AsyncGenerator
from typing import TypeVar

Output = TypeVar('Output', covariant=True)
Input = TypeVar('Input', contravariant=True)

type Fiber[Input, Output] = AsyncGenerator[Output, Input]
