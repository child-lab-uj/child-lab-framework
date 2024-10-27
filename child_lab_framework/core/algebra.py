from enum import IntEnum

import numpy as np

from ..typing.array import FloatArray1, FloatArray2, FloatArray3, FloatArray4


class Axis(IntEnum):
    X = 1
    Y = 2
    Z = 3


def rotation_matrix(angle: float, axis: Axis) -> FloatArray2:
    sin: float = np.sin(angle)
    cos: float = np.cos(angle)

    match axis:
        case Axis.X:
            return np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])

        case Axis.Y:
            return np.array(
                [[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]], dtype=np.float32
            )

        case Axis.Z:
            return np.array(
                [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32
            )


def normalized(vecs: FloatArray2) -> FloatArray2:
    norm = np.linalg.norm(vecs, ord=2.0, axis=1)
    return vecs / norm


def normalized_3d(batched_vecs: FloatArray3) -> FloatArray3:
    norm = np.linalg.norm(batched_vecs, ord=2.0, axis=2)[:, :, np.newaxis]
    return batched_vecs / norm


def orthogonal(vecs: FloatArray2) -> FloatArray2:
    return vecs[:, [1, 0]] * np.array([1.0, -1.0], dtype=np.float32)


def kabsch(
    points_input_frame: FloatArray4, points_output_frame: FloatArray4
) -> tuple[FloatArray2, FloatArray1]:
    assert points_input_frame.shape == points_output_frame.shape

    num_rows, num_cols = points_input_frame.shape
    if num_rows != 3:
        raise Exception(
            f'matrix points_input_frame is not 3xN, it is {num_rows}x{num_cols}'
        )

    num_rows, num_cols = points_output_frame.shape
    if num_rows != 3:
        raise Exception(
            f'matrix points_output_frame is not 3xN, it is {num_rows}x{num_cols}'
        )

    # find mean column wise
    centroid_A = np.mean(points_input_frame, axis=1)
    centroid_B = np.mean(points_output_frame, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = points_input_frame - centroid_A
    Bm = points_output_frame - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print('det(R) < 0, reflection detected!, correcting for it ...')
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t.squeeze()
