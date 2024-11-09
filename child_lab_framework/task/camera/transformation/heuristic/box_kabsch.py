from math import ceil, floor

import numpy as np

from .....core.algebra import kabsch
from .....core.calibration import Calibration
from .....core.transformation import EuclideanTransformation
from .....typing.array import FloatArray2, IntArray1
from .... import pose


def estimate(
    from_pose: pose.Result,
    to_pose: pose.Result,
    from_depth: FloatArray2,
    to_depth: FloatArray2,
    from_calibration: Calibration,
    to_calibration: Calibration,
    confidence_threshold: float,
) -> EuclideanTransformation | None:
    from_cloud = __cloud_from_bounding_boxes(
        from_pose, from_calibration, from_depth, confidence_threshold
    )

    if from_cloud is None:
        return None

    to_cloud = __cloud_from_bounding_boxes(
        to_pose, to_calibration, to_depth, confidence_threshold
    )

    if to_cloud is None:
        return None

    from_cloud, to_cloud = __truncate_to_equal_size(from_cloud, to_cloud)

    return EuclideanTransformation(*kabsch(from_cloud, to_cloud))


def __cloud_from_bounding_boxes(
    poses: pose.Result,
    calibration: Calibration,
    depth: FloatArray2,
    confidence_threshold: float,
) -> FloatArray2 | None:
    height, width = depth.shape
    cx, cy = calibration.optical_center
    fx, fy = calibration.focal_length

    space_chunks: list[FloatArray2] = []

    box: IntArray1
    for box in poses.boxes:
        if box[4] < confidence_threshold:
            continue

        x_start = max(int(floor(box[0])), 0)
        y_start = max(int(floor(box[1])), 0)
        x_end = min(int(ceil(box[2])), width)
        y_end = min(int(ceil(box[3])), height)

        x_indices, y_indices = np.meshgrid(
            np.arange(x_start, x_end, step=1.0, dtype=np.float32),
            np.arange(y_start, y_end, step=1.0, dtype=np.float32),
            indexing='xy',
        )

        z = depth[y_start:y_end, x_start:x_end]

        x = (x_indices - cx) * z / fx
        y = (y_indices - cy) * z / fy

        points = np.concatenate(
            (x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)),
            axis=1,
        )

        space_chunks.append(points)

    if len(space_chunks) == 0:
        return None

    return np.concatenate(space_chunks, axis=0, dtype=np.float32, casting='unsafe')


def __truncate_to_equal_size(
    points1: FloatArray2,
    points2: FloatArray2,
) -> tuple[FloatArray2, FloatArray2]:
    n_points1, _ = points1.shape
    n_points2, _ = points2.shape

    if n_points1 == n_points2:
        return points1, points2

    elif n_points1 < n_points2:
        mask = np.ones(n_points2, dtype=bool)
        mask[n_points1:] = False
        np.random.shuffle(mask)

        return points1, points2[mask]

    else:
        mask = np.ones(n_points1, dtype=bool)
        mask[n_points2:] = False
        np.random.shuffle(mask)

        return points1[mask], points2
