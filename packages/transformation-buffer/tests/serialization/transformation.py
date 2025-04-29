from serde.json import from_json, to_json
from syrupy.assertion import SnapshotAssertion
from transformation_buffer import Transformation


def test_serialize_identity_transformation_json(snapshot: SnapshotAssertion) -> None:
    transformation = Transformation.identity()
    serialized = to_json(transformation)
    assert serialized == snapshot


def test_deserialize_identity_transformation_json() -> None:
    data = """
        {
            "rotation_and_translation": {
                "data": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ],
                "dtype": "torch.float64"
            }
        }
    """

    deserialized = from_json(Transformation, data)
    expected = Transformation.identity()

    assert deserialized == expected
