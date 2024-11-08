import typing

import cv2
import numpy as np

from .....task import pose
from .....typing.array import (
    FloatArray1,
    FloatArray2,
    FloatArray3,
    IntArray1,
)


def common_points_indicator(
    points1: FloatArray2,
    points2: FloatArray2,
    confidence_threshold: float,
) -> IntArray1:
    sufficient_confidence = (
        np.minimum(points1.view()[:, -1], points2.view()[:, -1]) >= confidence_threshold
    )

    x_non_zero = (points1[:, 0] * points2[:, 0]) >= 0.0
    y_non_zero = (points1[:, 1] * points2[:, 1]) >= 0.0

    indicator = np.squeeze(np.where(sufficient_confidence & x_non_zero & y_non_zero))

    return typing.cast(IntArray1, indicator)


def estimate(
    from_pose: pose.Result,
    to_pose: pose.Result,
    from_depth: FloatArray2,
    to_depth: FloatArray2,
    intrinsics_matrix: FloatArray2,
    distortion: FloatArray1,
    confidence_threshold: float,
) -> tuple[FloatArray1, FloatArray1] | None:
    # NOTE: Workaround; fix when inter-camera actor recognition is introduced
    if len(from_pose.actors) != len(to_pose.actors):
        return None

    from_keypoints = from_pose.depersonificated_keypoints
    to_keypoints = to_pose.depersonificated_keypoints
    common = common_points_indicator(from_keypoints, to_keypoints, confidence_threshold)

    if common.size < 3:
        return None

    from_points_2d: FloatArray2 = from_keypoints.view()[common][:, [0, 1]]
    to_points_2d: FloatArray2 = to_keypoints.view()[common][:, [0, 1]]

    from_xs, from_ys = from_points_2d.astype(np.int32).T

    from_xs[from_xs >= 1920] = 1919
    from_ys[from_ys >= 1080] = 1079

    from_depths: FloatArray2 = from_depth[from_ys, from_xs].reshape(-1, 1)

    from_points_3d: FloatArray3 = np.concatenate((from_points_2d, from_depths), axis=1)

    success, rotation, translation = cv2.solvePnP(
        from_points_3d,
        to_points_2d,
        intrinsics_matrix,
        distortion,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_SQPNP,
    )

    if not success:
        return None

    return (typing.cast(FloatArray1, rotation), typing.cast(FloatArray1, translation))
