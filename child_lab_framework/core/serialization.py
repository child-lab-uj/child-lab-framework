from types import NoneType
from typing import Protocol, Self, TypeGuard

type Value = None | int | float | str | list[Value] | dict[str, Value]


class DeserializeError(Exception): ...


def assert_proper_serialization(data: dict[object, object]) -> None:
    for key, value in data.items():
        if not isinstance(key, str):
            raise DeserializeError(f'Non-string key found: {key}')

        if not is_allowed(value):
            raise DeserializeError(f'Illegal value found: {value}')


def is_allowed(value: object) -> TypeGuard[Value]:
    if type(value) in {NoneType, bool, int, float, str}:
        return True

    match value:
        case list(values):
            return all(map(is_allowed, values))

        case dict(entries):
            return is_serialization(entries)

        case _:
            return False


def is_serialization(data: dict[object, object]) -> TypeGuard[dict[str, Value]]:
    return all(isinstance(key, str) and is_allowed(value) for key, value in data.items())


class Serializable(Protocol):
    def serialize(self) -> dict[str, Value]: ...

    @classmethod
    def deserialize(cls, data: dict[str, Value]) -> Self: ...
