import typing
from collections.abc import Sequence


def imputed_with_closest_known_reference[T](batch: Sequence[T | None]) -> list[T] | None:
    if all(element is not None for element in batch):
        return typing.cast(list[T], list(batch))

    if all(element is None for element in batch):
        return None

    n = len(batch)
    i = 0

    imputed: list[T | None] = [None for _ in range(n)]
    closest_not_none: T | None = None

    while i < n:
        if (element := batch[i]) is not None:
            closest_not_none = element
            imputed[i] = element
        else:
            imputed[i] = closest_not_none

        i += 1

    if batch[0] is None:
        for i, closest_not_none in enumerate(batch):
            if closest_not_none is not None:
                break

        for j in range(i):
            imputed[j] = closest_not_none

    return typing.cast(list[T], imputed)
