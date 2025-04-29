import serde
import torch
from jaxtyping import Float
from serde.json import from_json, to_json
from syrupy.assertion import SnapshotAssertion
from transformation_buffer.serialization import tensor

tensor.init()


@serde.serde
class WrapperClass:
    tensor: torch.Tensor


@serde.serde
class WrapperClassWithJaxtyping:
    tensor: Float[torch.Tensor, '3 3'] = serde.field(
        serializer=tensor.serialize_tensor,
        deserializer=tensor.deserialize_tensor,
    )


def test_serialize_empty_tensor_json(snapshot: SnapshotAssertion) -> None:
    obj = WrapperClass(torch.tensor(()))
    serialized = to_json(obj)
    assert serialized == snapshot


def test_serialize_small_tensor_json(snapshot: SnapshotAssertion) -> None:
    tensor = torch.tensor(((1, 2, 3), (4, 5, 6), (7, 8, 9)), dtype=torch.float16)
    obj = WrapperClass(tensor)
    serialized = to_json(obj)
    assert serialized == snapshot


def test_serialize_small_tensor_jaxtyping_json(snapshot: SnapshotAssertion) -> None:
    tensor = torch.tensor(((1, 2, 3), (4, 5, 6), (7, 8, 9)), dtype=torch.float16)
    obj = WrapperClassWithJaxtyping(tensor)
    serialized = to_json(obj)
    assert serialized == snapshot


def test_deserialize_empty_tensor_json() -> None:
    data = """
        {
            "tensor": {
                "data": [],
                "dtype": "torch.float64"
            }
        }
    """

    deserialized = from_json(WrapperClass, data).tensor
    expected = torch.tensor((), dtype=torch.float64)

    assert torch.all(deserialized == expected)


def test_deserialize_small_tensor_json() -> None:
    data = """
        {
            "tensor": {
                "data": [
                    [1, 2, 3],
                    [4, 5, 6]
                ],
                "dtype": "torch.float16"
            }
        }
    """

    deserialized = from_json(WrapperClass, data).tensor
    expected = torch.tensor(((1, 2, 3), (4, 5, 6)), dtype=torch.float16)

    assert torch.all(deserialized == expected)
