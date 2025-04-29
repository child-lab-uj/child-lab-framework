from pathlib import Path

import pytest
from serde.json import from_json, to_json
from syrupy.assertion import SnapshotAssertion
from transformation_buffer import Buffer, Transformation


@pytest.fixture
def empty_buffer() -> Buffer[str]:
    return Buffer[str]()


@pytest.fixture
def buffer() -> Buffer[str]:
    return (
        Buffer[str]()
        .add_transformation('a', 'b', Transformation.identity())
        .add_transformation('a', 'c', Transformation.identity())
        .add_transformation('b', 'd', Transformation.identity())
        .add_transformation('d', 'e', Transformation.identity())
        .add_transformation('c', 'e', Transformation.identity())
        .add_transformation('d', 'f', Transformation.identity())
        .add_transformation('g', 'h', Transformation.identity())
    )


def test_serialize_empty_buffer_json(
    empty_buffer: Buffer[str],
    snapshot: SnapshotAssertion,
) -> None:
    serialized = to_json(empty_buffer)
    assert serialized == snapshot


def test_serialize_buffer_json(
    buffer: Buffer[str],
    snapshot: SnapshotAssertion,
) -> None:
    serialized = to_json(buffer)
    assert serialized == snapshot


def test_deserialize_buffer_json() -> None:
    data_location = Path(__file__).parent / 'data' / 'serialized_buffer.json'
    data = data_location.read_text()

    deserialized = from_json(Buffer[str], data)
    assert deserialized['a', 'b'] == Transformation.identity()
