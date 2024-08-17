from enum import IntEnum
import numpy as np

from ..typing.array import FloatArray2, FloatArray3


class Axis(IntEnum):
    X = 1
    Y = 2
    Z = 3


def rotation_matrix(angle: float, axis: Axis) -> FloatArray2:
    sin: float = np.sin(angle)
    cos: float = np.cos(angle)

    match axis:
        case Axis.X:
            return np.array([
                [1.0, 0.0, 0.0],
                [0.0, cos, -sin],
                [0.0, sin, cos]
            ])

        case Axis.Y:
            return np.array([
                [cos, 0.0, sin],
                [0.0, 1.0, 0.0],
                [-sin, 0.0, cos]
            ], dtype=np.float32)

        case Axis.Z:
            return np.array([
                [cos, -sin, 0.0],
                [sin, cos, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)


def normalized(vecs: FloatArray2) -> FloatArray2:
    norm = np.linalg.norm(vecs, ord=2.0, axis=0)
    return vecs / norm


def orthogonal(vecs: FloatArray2) -> FloatArray2:
    return vecs[:, 1::-1] * np.array([1.0, -1.0], dtype=np.float32)
