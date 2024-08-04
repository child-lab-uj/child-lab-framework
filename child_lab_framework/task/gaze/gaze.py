from dataclasses import dataclass
import numpy as np
from collections.abc import Generator
from typing import Literal, Iterable

from ...core.stream import autostart
from .. import pose, face
from ..pose.keypoint import YoloKeypoint
from ...typing.array import FloatArray1, FloatArray2
from ...typing.stream import Fiber

# Multi-camera gaze direction estimation without strict algebraic camera models:
# 1. estimate actor's skeleton on each frame in both cameras
#    (heuristic: adults' keypoints have higher variance, children have smaller bounding boxes)
# 2. compute a ceiling baseline vector (perpendicular to shoulder line in a ceiling camera)
# 3. detect actor's face on the other camera
# 4. compute the offset-baseline vector (normal to face, Wim's MediaPipe solution')
# 5. Rotate it to the celing camera's space and combine with the ceiling baseline


@dataclass
class Input:
    ceiling_poses: list[pose.Result | None]
    side_poses: list[pose.Result | None]
    faces: list[face.Result | None]


@dataclass
class Result:
    centres: FloatArray2
    versors: FloatArray2

    def iter(self) -> Iterable[tuple[FloatArray1, FloatArray1]]:
        return zip(self.centres, self.versors)


def orthogonal_normal(vecs: FloatArray2) -> FloatArray2:
    return vecs[:, 1::-1] * np.array([1.0, -1.0], dtype=np.float32)


class Estimator:
    def __init__(self) -> None:
        ...

    # TODO: calculate using pose eye keypoints and fallback to shoulders if eyes were not detected
    def __ceiling_baseline(self, result: pose.Result) -> Result:
        left_shoulder: FloatArray2 = result.keypoints[:, YoloKeypoint.LEFT_SHOULDER.value, :]
        right_shoulder: FloatArray2 = result.keypoints[:, YoloKeypoint.RIGHT_SHOULDER.value, :]

        centres: FloatArray2 = (left_shoulder + right_shoulder) / 2.0
        centres[:, -1] = left_shoulder[:, -1] * right_shoulder[:, -1]  # confidence of two keypoints as joint probability

        # convention: shoulder vector goes from left to right ->
        # versor (calculated as [y, -x]) points to the actor's front
        versors = orthogonal_normal(right_shoulder - left_shoulder)

        return Result(centres, versors)

    def __rotation_angles(self, ceiling: pose.Result, side: pose.Result) -> FloatArray1:
        ceiling_shoulders: FloatArray1 = ceiling.keypoints[0, YoloKeypoint.LEFT_SHOULDER, :] - ceiling.keypoints[0, YoloKeypoint.RIGHT_SHOULDER, :]
        side_shoulders: FloatArray1 = side.keypoints[0, YoloKeypoint.LEFT_SHOULDER, :] - side.keypoints[0, YoloKeypoint.RIGHT_SHOULDER, :]

        angles_from_shoulders_to_cameras = np.arccos(side_shoulders / ceiling_shoulders)

        return angles_from_shoulders_to_cameras * [1.0, 2.0] - np.pi / 2.0

    def __predict(
        self,
        ceiling_pose: pose.Result | None,
        side_pose: pose.Result | None,
        face: pose.Result | None
    ) -> Result:
        ...

    # TODO: receive other results too (face-based correction etc.)
    @autostart
    def stream(self) -> Fiber[Input | None, list[Result | None] | None]:
        results: list[Result | None] | None = None

        while True:
            match (yield results):
                case Input(ceiling_poses, side_poses, faces):
                    results = [
                        self.__ceiling_baseline(result)
                        if result is not None
                        else None
                        for result in ceiling_poses
                    ]

                case _:
                    results = None
