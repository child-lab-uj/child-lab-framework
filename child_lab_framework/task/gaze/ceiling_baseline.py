import numpy as np

from ...core.algebra import batch_normalized, orthogonal
from ...task import pose
from ...typing.array import FloatArray2
from ..pose.keypoint import YoloKeypoint


def estimate(
    poses: pose.Result,
    *,
    face_keypoint_threshold: float = 0.75,
) -> tuple[FloatArray2, FloatArray2]:
    left_shoulders: FloatArray2 = poses.keypoints[:, YoloKeypoint.LEFT_SHOULDER.value, :2]
    right_shoulders: FloatArray2 = poses.keypoints[
        :, YoloKeypoint.RIGHT_SHOULDER.value, :2
    ]

    # convention: shoulder vector goes from left to right -> versor (calculated as [y, -x]) points to the actor's front
    directions = batch_normalized(orthogonal(right_shoulders - left_shoulders))

    starts = np.zeros_like(directions)

    batched_face_keypoints = poses.keypoints[:, :5, :]

    face: FloatArray2
    for i, face in enumerate(batched_face_keypoints):
        confidences = face.view()[:, 2]

        if confidences[0] >= face_keypoint_threshold:
            starts[i, :] = face[0, :2]

        elif min(confidences[1], confidences[2]) >= face_keypoint_threshold:
            starts[i, :] = (face[1, :2] + face[2, :2]) / 2.0

        elif min(confidences[3], confidences[4]) >= face_keypoint_threshold:
            starts[i, :] = (face[3, :2] + face[4, :2]) / 2.0

        else:
            starts[i, :] = (left_shoulders[i, :] + right_shoulders[i, :]) / 2.0

    return starts, directions
