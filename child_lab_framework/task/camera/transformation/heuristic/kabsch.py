import typing

import numpy as np

from .....core.algebra import kabsch
from .....core.transformation import EuclideanTransformation
from .....typing.array import FloatArray2, IntArray1
from .... import pose


def estimate(
    from_pose: pose.Result,
    from_pose_3d: pose.Result3d,
    to_pose: pose.Result,
    to_pose_3d: pose.Result3d,
    confidence_threshold: float,
) -> EuclideanTransformation | None:
    from_keypoints = from_pose.flat_points_with_confidence
    to_keypoints = to_pose.flat_points_with_confidence

    common = __common_points_indicator(from_keypoints, to_keypoints, confidence_threshold)

    if common.size < 6:
        return None

    from_points_3d = from_pose_3d.flat_points[common]
    to_points_3d = to_pose_3d.flat_points[common]

    return EuclideanTransformation(*kabsch(from_points_3d, to_points_3d))


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
