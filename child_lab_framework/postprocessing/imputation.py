import typing
from collections.abc import Sequence

import numpy as np
from more_itertools import first_true


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


def imputed_with_zeros_reference[Shape: tuple[int, ...], Type: np.generic](
    batch: Sequence[np.ndarray[Shape, np.dtype[Type]] | None],
) -> list[np.ndarray[Shape, np.dtype[Type]]] | None:
    first_not_none = first_true(batch, None, lambda x: x is not None)

    if first_not_none is None:
        return None

    fill_element = np.zeros_like(first_not_none)

    return [element if element is not None else fill_element for element in batch]
