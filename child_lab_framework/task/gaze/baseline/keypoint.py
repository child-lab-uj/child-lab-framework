import typing
from dataclasses import dataclass

import cv2
import numpy as np

from ....core.algebra import normalized, orthogonal
from ....core.video import Properties
from ....typing.array import FloatArray2, IntArray1
from ....typing.video import Frame
from ... import pose, visualization
from ...pose.keypoint import YoloKeypoint


@dataclass(frozen=True, slots=True)
class Result:
    centers: FloatArray2
    directions: FloatArray2

    def visualize(
        self,
        frame: Frame,
        frame_properties: Properties,
        configuration: visualization.Configuration,
    ) -> Frame:
        starts = self.centers
        ends = starts + float(configuration.gaze_line_length) * self.directions

        start: IntArray1
        end: IntArray1

        color = configuration.gaze_baseline_line_color
        thickness = configuration.gaze_baseline_line_thickness

        for start, end in zip(
            starts.astype(np.int32),
            ends.astype(np.int32),
        ):
            cv2.circle(
                frame,
                typing.cast(cv2.typing.Point, start),
                3,
                (0.0, 0.0, 255.0, 1.0),
                1,
            )

            cv2.line(
                frame,
                typing.cast(cv2.typing.Point, start),
                typing.cast(cv2.typing.Point, end),
                color,
                thickness,
            )

        return frame


def estimate(
    poses: pose.Result,
    *,
    face_keypoint_threshold: float = 0.75,
) -> Result:
    left_shoulders: FloatArray2 = poses.keypoints[:, YoloKeypoint.LEFT_SHOULDER.value, :2]
    right_shoulders: FloatArray2 = poses.keypoints[
        :, YoloKeypoint.RIGHT_SHOULDER.value, :2
    ]

    # convention: shoulder vector goes from left to right -> versor (calculated as [y, -x]) points to the actor's front
    directions = normalized(orthogonal(right_shoulders - left_shoulders))

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

    return Result(starts, directions)
