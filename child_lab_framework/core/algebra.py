import typing
from enum import IntEnum
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

from ..typing.array import FloatArray1, FloatArray2, FloatArray3
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


def euler_angles_from_rotation_matrix(
    rotation: FloatArray2,
) -> np.ndarray[tuple[Literal[3]], np.dtype[np.float32]]:
    return (  # type: ignore[no-any-return]
        Rotation.from_matrix(rotation).as_euler('xyz', degrees=False).astype(np.float32)
    )


def rotation_matrix_from_euler_angles(
    angles: np.ndarray[tuple[Literal[3]], np.dtype[np.float32]],
) -> FloatArray2:
    return Rotation.from_euler('xyz', angles, degrees=False).as_matrix()  # type: ignore[no-any-return]


def normalized(vecs: FloatArray2) -> FloatArray2:
    norm: FloatArray1 = np.linalg.norm(vecs, ord=2.0, axis=1)
    return vecs / norm


def normalized_3d(batched_vecs: FloatArray3) -> FloatArray3:
    norm: FloatArray3 = np.linalg.norm(batched_vecs, ord=2.0, axis=2)[:, :, np.newaxis]
    return batched_vecs / norm


def orthogonal(vecs: FloatArray2) -> FloatArray2:
    return vecs[:, [1, 0]] * np.array([1.0, -1.0], dtype=np.float32)


def kabsch(
    from_points: FloatArray2,
    to_points: FloatArray2,
) -> tuple[FloatArray2, FloatArray1]:
    """
    Find an affine transformation given by a `3 x 3` rotation matrix `R` and a translation vector `t` such that:\\
    `to_points = R @ from_points + t`.

    Parameters
    ---
    from_points: FloatArray2
        An `n x 3` array of 3D points to transform into `to_points` by applying the transformation.

    to_points: FloatArray2
        An `n x 3` array of 3D points to obtain from `from_points` by applying the transformation.

    Returns
    ---
    result: tuple[FloatArray2, FloatArray1]
        A `3 x 3` rotation matrix and `1 x 3` translation vector.
    """

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

    rotation = typing.cast(FloatArray2, v @ ut)

    # special reflection case
    if np.linalg.det(rotation) < 0.0:
        v[:, -1] *= -1.0
        rotation = typing.cast(FloatArray2, v @ ut)

    translation: FloatArray2 = -rotation @ from_center + to_center

    return rotation, translation.squeeze()


def depth_to_perspective(depth: FloatArray2, calibration: Calibration) -> FloatArray3:
    """
    Unproject the depth map into a perspective view.

    Parameters
    ---
    depth: FloatArray2
        A `height x width` depth map to unproject of shape.

    calibration: Calibration
        A calibration of the camera that observes the `depth`.

    Returns
    ---
    result: FloatArray3
        A `height x width x 3` array of 3D points in a camera perspective.
    """

    height, width = depth.shape

    fx, fy = calibration.focal_length
    cx, cy = calibration.optical_center

    xs, ys = np.meshgrid(np.arange(width), np.arange(height))

    x = np.expand_dims((xs - cx) * depth / fx, axis=-1)
    y = np.expand_dims((ys - cy) * depth / fy, axis=-1)
    z = np.expand_dims(depth, axis=-1)

    return np.concatenate((x, y, z), axis=-1)


def point_cloud(
    rgb: FloatArray3,
    depth: FloatArray2,
    /,
    calibration: Calibration,
) -> FloatArray3:
    """
    Transform an `rgb` image and its corresponding depth map `depth` into a point cloud seen in a perspective.

    Parameters
    ---
    rgb: FloatArray3
        A `height x width x 3` representation of an RGB image seen from a camera.

    depth: FloatArray2
        A `height x width` depth map corresponding to the `rgb` image.

    calibration: Calibration
        A calibration of the camera that observes the `rgb` image and its `depth` depth map.

    Returns
    ---
    point_cloud: FloarArray3
        A `height x width x 6` array of vectors: `[x, y, z, r, g, b]` representing perspective points with their corresponding RGB values.
    """

    return np.concatenate((depth_to_perspective(depth, calibration), rgb), axis=-1)
