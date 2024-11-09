import typing

import cv2
import numpy as np

from .....core.algebra import (
    euler_angles_from_rotation_matrix,
    rotation_matrix_from_euler_angles,
)
from .....core.calibration import Calibration
from .....core.transformation import ProjectiveTransformation
from .....task import pose
from .....typing.array import FloatArray1, FloatArray2, IntArray1


def estimate(
    from_pose: pose.Result,
    from_pose_3d: pose.Result3d,
    to_pose: pose.Result,
    to_pose_3d: pose.Result3d,
    from_calibration: Calibration,
    to_calibration: Calibration,
    confidence_threshold: float,
) -> ProjectiveTransformation | None:
    from_points_2d = from_pose.flat_points_with_confidence
    to_points_2d = to_pose.flat_points_with_confidence

    common = __common_points_indicator(from_points_2d, to_points_2d, confidence_threshold)

    if common.size < 6:
        return None

    from_points_2d = from_points_2d[common][:, [0, 1]]
    to_points_2d = to_points_2d[common][:, [0, 1]]

    to_points_3d = to_pose_3d.flat_points[common]
    from_points_3d = from_pose_3d.flat_points[common]

    success, rotation, translation = cv2.solvePnP(
        from_points_3d,
        to_points_2d,
        to_calibration.intrinsics,
        to_calibration.distortion,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_SQPNP,
    )

    if not success:
        return None

    success, inverse_rotation, inverse_translation = cv2.solvePnP(
        to_points_3d,
        from_points_2d,
        from_calibration.intrinsics,
        from_calibration.distortion,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_SQPNP,
    )

    if not success:
        return None

    rotation = typing.cast(FloatArray2, cv2.Rodrigues(rotation)[0])
    inverse_rotation = typing.cast(FloatArray2, cv2.Rodrigues(inverse_rotation)[0])
    inverse_inverse_rotation = np.linalg.inv(inverse_rotation)

    average_rotation = rotation_matrix_from_euler_angles(
        (
            euler_angles_from_rotation_matrix(rotation)
            + euler_angles_from_rotation_matrix(inverse_inverse_rotation)
        )
        / 2.0
    )

    average_translation: FloatArray1 = (translation - inverse_translation) / 2.0

    return ProjectiveTransformation(
        average_rotation,
        average_translation,
        to_calibration,
    )


def __common_points_indicator(
    points1: FloatArray2,
    points2: FloatArray2,
    confidence_threshold: float,
) -> IntArray1:
    sufficient_confidence = (
        np.minimum(points1.view()[:, 2], points2.view()[:, 2]) >= confidence_threshold
    )

    x_non_zero = (points1[:, 0] * points2[:, 0]) > 0.0
    y_non_zero = (points1[:, 1] * points2[:, 1]) > 0.0

    indicator = np.squeeze(np.where(sufficient_confidence & x_non_zero & y_non_zero))

    return typing.cast(IntArray1, indicator)
