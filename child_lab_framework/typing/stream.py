from typing import TypeVar
from collections.abc import Generator, AsyncGenerator

Output = TypeVar("Output", covariant=True)
Input = TypeVar("Input", contravariant=True)

type SyncFiber[Input, Output] = Generator[Output, Input, None]
type Fiber[Input, Output] = AsyncGenerator[Output, Input]
