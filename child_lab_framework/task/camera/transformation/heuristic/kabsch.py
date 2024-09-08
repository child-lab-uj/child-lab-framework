import numpy as np
import typing

from .... import pose
from .....typing.array import IntArray1, FloatArray1, FloatArray2, FloatArray3


def common_points_indicator(points1: FloatArray2, points2: FloatArray2, confidence_threshold: float) -> IntArray1:
    probabilities = np.minimum(
        points1.view()[:, -1],
        points2.view()[:, -1]
    )

    indicator = np.squeeze(np.where(probabilities >= confidence_threshold))

    return typing.cast(IntArray1, indicator)


def estimate(
    from_pose: pose.Result,
    to_pose: pose.Result,
    from_depth: FloatArray2,
    to_depth: FloatArray2,
    confidence_threshold: float
) -> tuple[FloatArray2, FloatArray1] | None:
    # NOTE: A quick workaround
    if len(from_pose.actors) != len(to_pose.actors):
        return None

    from_keypoints = from_pose.depersonificated_keypoints
    to_keypoints = to_pose.depersonificated_keypoints
    common = common_points_indicator(from_keypoints, to_keypoints, confidence_threshold)

    from_points_2d: FloatArray2 = from_keypoints.view()[common][:, [0, 1]]
    to_points_2d: FloatArray2 = to_keypoints.view()[common][:, [0, 1]]

    from_xs, from_ys = from_points_2d.astype(np.int32).T
    to_xs, to_ys = to_points_2d.astype(np.int32).T

    from_depths: FloatArray2 = from_depth[from_ys, from_xs, np.newaxis]
    to_depths: FloatArray2 = to_depth[to_ys, to_xs, np.newaxis]

    from_points_3d: FloatArray3 = np.concatenate((from_points_2d, from_depths), axis=1)
    to_points_3d: FloatArray3 = np.concatenate((to_points_2d, to_depths), axis=1)

    from_center: FloatArray1 = typing.cast(FloatArray1, np.mean(from_points_3d, axis=0))
    to_center: FloatArray1 = typing.cast(FloatArray1, np.mean(to_points_3d, axis=0))

    translation: FloatArray1 = to_center - from_center

    cross_covariance: FloatArray2 = np.dot(
        (from_points_3d - from_center).T,
        to_points_3d - to_center
    )

    u, _, vt = np.linalg.svd(cross_covariance)
    v = vt.T
    ut = u.T

    if np.linalg.det(np.dot(v, ut)) < 0.0:
        v[-1, :] *= -1.0

    rotation: FloatArray2 = np.dot(v, ut)

    return rotation, translation
