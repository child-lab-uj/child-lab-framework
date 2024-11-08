from enum import IntEnum

import numpy as np

from ..typing.array import FloatArray1, FloatArray2, FloatArray3, FloatArray6
from .calibration import Calibration


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
    from_points: FloatArray2,
    to_points: FloatArray2,
) -> tuple[FloatArray2, FloatArray1]:
    if from_points.shape != to_points.shape:
        raise ValueError(
            f'Expected inputs and outputs of equal shape, got input: {from_points.shape}, output: {to_points.shape}'
        )

    n_rows, n_columns = from_points.shape
    if n_columns != 3:
        raise ValueError(
            f'Expected points_input_frame to have shape n x 3, got {n_rows} x {n_columns}'
        )

    n_rows, n_columns = to_points.shape
    if n_columns != 3:
        raise ValueError(
            f'Expected points_output_frame to have shape n x 3, got {n_rows} x {n_columns}'
        )

    from_center = np.mean(from_points, axis=0)
    to_center = np.mean(to_points, axis=0)

    from_points_centered = from_points - from_center
    to_points_centered = to_points - to_center

    cross_covariance = from_points_centered.T @ to_points_centered

    decomposition = np.linalg.svd(cross_covariance)
    ut: FloatArray2 = decomposition.U.T
    v: FloatArray2 = decomposition.Vh.T

    rotation: FloatArray2 = v @ ut

    # special reflection case
    if np.linalg.det(rotation) < 0.0:
        v[:, -1] *= -1.0
        rotation: FloatArray2 = v @ ut

    translation: FloatArray2 = -rotation @ from_center + to_center

    return rotation, translation.squeeze()


def make_point_cloud(
    rgb: FloatArray3,
    depth: FloatArray2,
    calibration: Calibration,
) -> FloatArray6:
    height, width, *_ = rgb.shape

    fx, fy = calibration.focal_length
    cx, cy = calibration.optical_center

    xs, ys = np.meshgrid(np.arange(width), np.arange(height))

    x = np.expand_dims((xs - cx) * depth / fx, axis=-1)
    y = np.expand_dims((ys - cy) * depth / fy, axis=-1)
    z = np.expand_dims(depth, axis=-1)

    return np.concatenate((x, y, z, rgb), axis=-1)
