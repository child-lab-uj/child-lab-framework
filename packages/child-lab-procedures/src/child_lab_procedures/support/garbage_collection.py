import gc
from contextlib import ContextDecorator
from types import TracebackType
from typing import Literal, Self


class no_garbage_collection(ContextDecorator):
    """
    Disable garbage collector.
    """

    def __init__(self, *_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__()

    def __enter__(self) -> Self:
        gc.disable()
        return self

    def __exit__(
        self,
        _exception_kind: type | None,
        exception: Exception | None,
        _traceback: TracebackType | None,
        **_: object,
    ) -> Literal[False]:
        gc.enable()

        if exception is not None:
            raise exception

        return False
