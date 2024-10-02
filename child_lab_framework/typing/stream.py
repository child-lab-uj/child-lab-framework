from collections.abc import AsyncGenerator, Generator
from typing import TypeVar

Output = TypeVar('Output', covariant=True)
Input = TypeVar('Input', contravariant=True)

type SyncFiber[Input, Output] = Generator[Output, Input, None]
type Fiber[Input, Output] = AsyncGenerator[Output, Input]
