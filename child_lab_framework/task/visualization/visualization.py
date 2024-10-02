import typing
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from ...core.stream import autostart
from ...core.video import Frame
from ...typing.stream import Fiber
from .. import gaze, pose
from ..pose.keypoint import YOLO_SKELETON

type Input = tuple[
    list[Frame] | None, list[pose.Result | None] | None, list[gaze.Result | None] | None
]


class Visualizer:
    BONE_COLOR = (255, 0, 0)
    BONE_THICKNESS = 2

    GAZE_COLOR = (0, 0, 255)
    GAZE_THICKNESS = 3

    JOINT_RADIUS = 5
    JOINT_COLOR = (0, 255, 0)

    confidence_threshold: float

    executor: ThreadPoolExecutor

    def __init__(
        self, executor: ThreadPoolExecutor, *, confidence_threshold: float
    ) -> None:
        self.confidence_threshold = confidence_threshold

    def __draw_skeleton_and_joints(self, frame: Frame, result: pose.Result) -> Frame:
        annotated_frame = frame.copy()

        for _, _, keypoints in result.iter():
            for i, j in YOLO_SKELETON:
                if keypoints[i, -1] < self.confidence_threshold:
                    continue

                if keypoints[j, -1] < self.confidence_threshold:
                    continue

                start = typing.cast(cv2.typing.Point, keypoints[i, :-1].astype(int))
                end = typing.cast(cv2.typing.Point, keypoints[j, :-1].astype(int))

                cv2.line(
                    annotated_frame, start, end, self.BONE_COLOR, self.BONE_THICKNESS
                )

            for keypoint in keypoints:
                if keypoint[-1] < self.confidence_threshold:
                    continue

                keypoint = typing.cast(cv2.typing.Point, keypoint.astype(np.int32))

                cv2.circle(
                    annotated_frame,
                    keypoint[:-1],
                    self.JOINT_RADIUS,
                    self.JOINT_COLOR,
                    -1,
                )

        return annotated_frame

    def __draw_gaze_estimation(self, frame: Frame, result: gaze.Result) -> Frame:
        annotated_frame = frame.copy()

        for centre, versor in result.iter():
            start = typing.cast(cv2.typing.Point, centre[[0, 1]].astype(int))
            end = typing.cast(
                cv2.typing.Point, (centre[[0, 1]] + 100.0 * versor).astype(int)
            )

            cv2.line(annotated_frame, start, end, self.GAZE_COLOR, self.GAZE_THICKNESS)

        return annotated_frame

    @autostart
    async def stream(self) -> Fiber[Input, list[Frame] | None]:
        annotated_frames: list[Frame] | None = None

        while True:
            match (yield annotated_frames):
                case list(frames), list(poses), list(gazes):
                    annotated_frames = []

                    for frame, pose, gaze in zip(frames, poses, gazes):
                        out = frame

                        if pose is not None:
                            out = self.__draw_skeleton_and_joints(out, pose)

                        if gaze is not None:
                            out = self.__draw_gaze_estimation(out, gaze)

                        annotated_frames.append(out)

                case list(frames), *_:
                    annotated_frames = frames

                case _:
                    annotated_frames = None
