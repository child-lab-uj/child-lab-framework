import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import IO, Literal, Protocol

import yaml

from . import serialization

# TODO: add file (and thread?) locks
# TODO: Add TOML


class Format(Enum):
    YAML = 'YAML'
    JSON = 'JSON'

    @staticmethod
    def guess(path: Path) -> 'Format | None':
        match path.suffix:
            case '.yml':
                return Format.YAML

            case '.json':
                return Format.JSON

            case _:
                return None


class Mode(Enum):
    READ = 'r'
    READ_WRITE = 'rw'
    WRITE_CREATE = 'w+'

    def __repr__(self) -> str:
        match self:
            case Mode.READ:
                return 'read-only'

            case Mode.READ_WRITE:
                return 'read-write'

            case Mode.WRITE_CREATE:
                return 'write-create'


def save(
    item: serialization.Serializable,
    path: Path,
    format: Format | None = None,
) -> None:
    format = format or Format.guess(path)

    if format is None:
        raise ValueError(f'Cannot infer output format of {path}')

    with File(path, Mode.WRITE_CREATE, format) as file:
        file.write(item.serialize())


def load[T: serialization.Serializable](
    result_type: type[T],
    path: Path,
    format: Format | None = None,
) -> T:
    format = format or Format.guess(path)

    if format is None:
        raise ValueError(f'Cannot infer input format of {path}')

    with File(path, Mode.READ, format) as file:
        return result_type.deserialize(file.read())

    raise RuntimeError()


class FileGuard(Protocol):
    def read(self) -> dict[str, serialization.Value]: ...
    def write(self, data: dict[str, serialization.Value]) -> None: ...


class File:
    location: Path
    format: Format
    mode: Mode

    _io: IO[str] | None

    def __init__(self, location: Path, mode: Mode, format: Format) -> None:
        self.location = location
        self.mode = mode
        self.format = format
        self._io = None

    def __enter__(self) -> FileGuard:
        self._io = io = open(self.location, self.mode.value)

        match self.format:
            case Format.YAML:
                return Yaml(io)

            case Format.JSON:
                return Json(io)

    def __exit__(
        self,
        _exception_kind: type[Exception] | None,
        _exception: Exception | None,
        traceback: TracebackType | None,
        **_: object,
    ) -> Literal[False]:
        if self._io is not None:
            self._io.close()

        return False

    def __repr__(self) -> str:
        return ' '.join(
            (
                repr(self.mode).title(),
                self.format.value,
                'file at',
                str(self.location),
            )
        )


@dataclass
class Yaml:
    _io: IO[str]

    def read(self) -> dict[str, serialization.Value]:
        data = yaml.safe_load(self._io)  # type: ignore[no-untyped-call]

        # TODO: use the type guard explicitly to not confuse the type checker
        serialization.assert_proper_serialization(data)

        return data  # type: ignore[no-any-return]  # guaranteed to contain valid value

    def write(self, data: dict[str, serialization.Value]) -> None:
        yaml.safe_dump(data, self._io)  # type: ignore[no-untyped-call]


@dataclass
class Json:
    _io: IO[str]

    def read(self) -> dict[str, serialization.Value]:
        data = json.load(self._io)

        # TODO: use the type guard explicitly to not confuse the type checker
        serialization.assert_proper_serialization(data)

        return data  # type: ignore[no-any-return]  # guaranteed to contain valid value

    def write(self, data: dict[str, serialization.Value]) -> None:
        json.dump(data, self._io)
