from collections.abc import Sequence

from more_itertools import first_true

from .interface import Interpolable


def interpolated[T: Interpolable](batch: Sequence[T | None]) -> list[T] | None:
    match first_true(batch, None, lambda x: x is not None):
        case None:
            return None

        case item:
            return item.__class__.interpolated(batch)
