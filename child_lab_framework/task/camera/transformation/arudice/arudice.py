import numpy as np
from .....typing.array import FloatArray1, FloatArray2


def create_transforms_from_arudie_face_0(
    arudie_size: float,
) -> list[tuple[FloatArray2, FloatArray1]]:
    # https://dugas.ch/transform_viewer/multi.html
    # Returns transforms from arudie face 0 to subsequent faces i.e. list[1] -> to face 1
    half = arudie_size / 2
    full = arudie_size

    return [
        # 0 -> identity
        (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)),
        # 1
        (
            np.array(
                [
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0],
                ],
                dtype=np.float32,
            ),
            np.array([0, half, -half], dtype=np.float32),
        )
        # 2
        (
            np.array(
                [
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1],
                ],
                dtype=np.float32,
            ),
            np.array([0, 0, -full], dtype=np.float32),
        )
        # 3
        (
            np.array(
                [
                    [1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0],
                ],
                dtype=np.float32,
            ),
            np.array([0, -half, -half], dtype=np.float32),
        )
        # 4
        (
            np.array(
                [
                    [0, 0, 1],
                    [0, 1, 0],
                    [-1, 0, 0],
                ],
                dtype=np.float32,
            ),
            np.array([half, 0, -half], dtype=np.float32),
        )
        # 5
        (
            np.array(
                [
                    [0, 0, -1],
                    [0, 1, 0],
                    [1, 0, 0],
                ],
                dtype=np.float32,
            ),
            np.array([-half, 0, -half], dtype=np.float32),
        ),
    ]
