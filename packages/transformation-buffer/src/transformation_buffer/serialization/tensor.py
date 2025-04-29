from typing import Any

import serde
import torch
from plum import dispatch


def init() -> None:
    serde.add_serializer(Serializer())
    serde.add_deserializer(Deserializer())


class Serializer:
    @dispatch
    def serialize(self, value: torch.Tensor) -> dict[str, Any]:
        return serialize_tensor(value)


class Deserializer:
    @dispatch
    def deserialize(self, cls: type[torch.Tensor], value: Any) -> torch.Tensor:
        return deserialize_tensor(value)


def serialize_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    return {'data': tensor.tolist(), 'dtype': str(tensor.dtype)}


def deserialize_tensor(data: Any) -> torch.Tensor:
    match data:
        case {'data': list(data), 'dtype': str(serialized_dtype)}:
            dtype: torch.dtype = eval(serialized_dtype, None, {'torch': torch})
            return torch.tensor(data, dtype=dtype)

        case _:
            raise serde.SerdeError('')
