import typing
import numpy as np

from ...core.sequence import imputed_with_reference_inplace
from ...core.algebra import orthogonal, normalized
from ...typing.array import FloatArray1, FloatArray2, FloatArray3
from ...task import pose, face
from ..pose.keypoint import YoloKeypoint


# TODO: Move the geometry part to task.camera.transformation.heuristic
def estimate(poses: list[pose.Result | None], faces: list[face.Result | None] | None) -> tuple[FloatArray3, FloatArray3]:
    result_centres: list[FloatArray2 | None] = []
    result_vectors: list[FloatArray2 | None] = []

    for pose, _ in zip(poses, faces or poses):  # TODO: use face detection
        if pose is None:
            result_centres.append(None)
            result_vectors.append(None)
            continue

        left_shoulder: FloatArray2 = pose.keypoints[:, YoloKeypoint.LEFT_SHOULDER.value, :]
        right_shoulder: FloatArray2 = pose.keypoints[:, YoloKeypoint.RIGHT_SHOULDER.value, :]

        centres: FloatArray2 = (left_shoulder + right_shoulder) / 2.0
        centres[:, -1] = left_shoulder[:, -1] * right_shoulder[:, -1]  # confidence of two keypoints as joint probability

        # convention: shoulder vector goes from left to right -> versor (calculated as [y, -x]) points to the actor's front
        vectors = normalized(orthogonal(right_shoulder - left_shoulder))

        result_centres.append(centres)
        result_vectors.append(vectors)

    return (
        np.stack(imputed_with_reference_inplace(result_centres)),
        np.stack(imputed_with_reference_inplace(result_vectors)),
    )
