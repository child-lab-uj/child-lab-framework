import logging
from itertools import count
from pathlib import Path

import torch
import vpc
from beartype import beartype

from child_lab_data.serde.result_tensor import deserialize, serialize

type Readable = vpc.pose.Result | vpc.pose.Result3d | vpc.gaze.Result3d
type Writeable = vpc.pose.Result | vpc.pose.Result3d | vpc.gaze.Result3d


class Reader[T: Readable]:
    ty: type[T]
    directory: Path
    counter: 'count[int]'

    def __init__(self, ty: type[T], directory: Path) -> None:
        self.ty = ty
        self.directory = directory
        self.counter = count()

    def skip(self, count: int) -> None:
        for _ in range(count):
            next(self.counter)

    @beartype
    def read(self, id: int | None = None) -> T | None:
        if id is None:
            id = next(self.counter)

        path = self.directory / f'{id}.pt'
        if not path.is_file():
            logging.warning(f'Serialized result not found at {path}')
            return None

        tensor: torch.Tensor = torch.load(path, weights_only=True)
        return deserialize(self.ty, tensor)


class Writer[T: Writeable]:
    ty: type[T]
    directory: Path
    counter: 'count[int]'

    def __init__(self, ty: type[T], directory: Path) -> None:
        self.ty = ty
        self.directory = directory
        self.counter = count()

    def skip(self, count: int) -> None:
        for _ in range(count):
            next(self.counter)

    @beartype
    def write(self, item: T, id: int | None = None) -> None:
        if id is None:
            id = next(self.counter)

        path = self.directory / f'{id}.pt'
        if path.is_file():
            logging.warning(f'Overwriting the existing result at {path}')

        tensor = serialize(item)
        torch.save(tensor, path)
