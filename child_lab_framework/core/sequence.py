import typing
import numpy as np


def imputed_with_reference_inplace[T](sequence: list[T | None]) -> list[T]:
    if all([element is not None for element in sequence]):
        return typing.cast(list[T], sequence)

    n = len(sequence)
    i = 0

    closest_not_none: T | None = None

    while i < n:
        if (element := sequence[i]) is not None:
            closest_not_none = element
        else:
            sequence[i] = closest_not_none

        i += 1

    if sequence[0] is None:
        for i, closest_not_none in enumerate(sequence):
            if closest_not_none is not None:
                break

        for j in range(i):
            sequence[j] = closest_not_none

    return typing.cast(list[T], sequence)


def imputed_with_zeros_reference_inplace[Size, Type](
    sequence: list[np.ndarray[Size, np.dtype] | None]
) -> list[np.ndarray[Size, np.dtype]] | None:
    first_not_none = None

    for element in sequence:
        if element is not None:
            first_not_none = element
            break

    if first_not_none is None:
        return None

    fill_element = np.zeros_like(first_not_none)

    for i in range(len(sequence)):
        if sequence[i] is None:
            sequence[i] = fill_element

    return typing.cast(list[np.ndarray], sequence)
