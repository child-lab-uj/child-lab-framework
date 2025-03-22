import pytest
import torch
from serde.json import from_json, to_json
from syrupy.assertion import SnapshotAssertion
from transformation_buffer.buffer import Buffer
from transformation_buffer.transformation import Transformation

IDENTITY = Transformation(
    torch.tensor(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=torch.float32,
    )
)


@pytest.mark.skip(reason='Serialization is not available yet.')
def test_serialize_empty_buffer(snapshot: SnapshotAssertion) -> None:
    buffer = Buffer[str]([])
    serialized = to_json(buffer)
    assert serialized == snapshot()


@pytest.mark.skip(reason='Serialization is not available yet.')
def test_serialize_simple_buffer(snapshot: SnapshotAssertion) -> None:
    buffer = Buffer(['a', 'b'])
    buffer['a', 'b'] = IDENTITY

    serialized = to_json(buffer)
    assert serialized == snapshot


@pytest.mark.skip(reason='Serialization is not available yet.')
def test_deserialize_simple_buffer() -> None:
    buffer = Buffer(['a', 'b'])
    buffer['a', 'b'] = IDENTITY

    serialized = to_json(buffer)
    deserialized = from_json(Buffer[str], serialized)

    assert deserialized.frames_of_reference == buffer.frames_of_reference
    assert deserialized['a', 'b'] == buffer['a', 'b']


@pytest.mark.skip(reason='Serialization is not available yet.')
def test_serialize_transformation(snapshot: SnapshotAssertion) -> None:
    serialized = to_json(IDENTITY)
    assert serialized == snapshot


@pytest.mark.skip(reason='Serialization is not available yet.')
def test_deserialize_transformation() -> None:
    serialized = to_json(IDENTITY)
    deserialized = from_json(Transformation, serialized)

    assert deserialized == IDENTITY
