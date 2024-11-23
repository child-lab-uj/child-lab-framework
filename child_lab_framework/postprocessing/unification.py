import typing
from collections.abc import Callable, Sequence

from .interface import Comparable


def substituted_with_minimum_reference[U, V: Comparable](
    batch: Sequence[U | None],
    key: Callable[[U], V],
) -> list[U] | None:
    if len(batch) == 0:
        return typing.cast(list[U], [])

    if all(element is None for element in batch):
        return None

    minimum = min(filter(None, batch), key=key)

    return [minimum for _ in batch]


def substituted_with_maximum_reference[U, V: Comparable](
    batch: Sequence[U | None],
    key: Callable[[U], V],
) -> list[U] | None:
    if len(batch) == 0:
        return typing.cast(list[U], [])

    if all(element is None for element in batch):
        return None

    minimum = max(filter(None, batch), key=key)

    return [minimum for _ in batch]
