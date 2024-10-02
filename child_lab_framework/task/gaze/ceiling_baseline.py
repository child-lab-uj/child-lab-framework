from itertools import repeat

import numpy as np

from ...core.algebra import orthogonal
from ...core.sequence import imputed_with_reference_inplace
from ...task import face, pose
from ...typing.array import FloatArray2, FloatArray3
from ..pose.keypoint import YoloKeypoint


def estimate(
    poses: list[pose.Result | None], faces: list[face.Result | None] | None
) -> tuple[FloatArray3, FloatArray3] | None:
    result_centres: list[FloatArray2 | None] = []
    result_vectors: list[FloatArray2 | None] = []

    for frame_poses, _ in zip(poses, faces or repeat(None)):  # TODO: use face detection
        if frame_poses is None or len(frame_poses.actors) < 2:  # NOTE: workaround
            result_centres.append(None)
            result_vectors.append(None)
            continue

        left_shoulder: FloatArray2 = frame_poses.keypoints[
            :, YoloKeypoint.LEFT_SHOULDER.value, :
        ]
        right_shoulder: FloatArray2 = frame_poses.keypoints[
            :, YoloKeypoint.RIGHT_SHOULDER.value, :
        ]

        centres: FloatArray2 = (left_shoulder + right_shoulder) / 2.0
        centres[:, -1] = (
            left_shoulder[:, -1] * right_shoulder[:, -1]
        )  # confidence of two keypoints as joint probability

        # convention: shoulder vector goes from left to right -> versor (calculated as [y, -x]) points to the actor's front
        vectors = orthogonal(right_shoulder - left_shoulder)

        result_centres.append(centres)
        result_vectors.append(vectors)

    match (
        imputed_with_reference_inplace(result_centres),
        imputed_with_reference_inplace(result_vectors),
    ):
        case list(imputed_centres), list(imputed_vectors):
            return np.stack(imputed_centres), np.stack(imputed_vectors)

        case _:
            return None
