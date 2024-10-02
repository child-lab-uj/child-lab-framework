import typing
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat, starmap

import cv2
import numpy as np

from ...core.stream import autostart
from ...core.video import Frame, Properties
from ...typing.array import FloatArray1
from ...typing.stream import Fiber
from .. import pose
from ..gaze import ceiling_projection
from ..pose.keypoint import YOLO_SKELETON

type Input = tuple[
    list[Frame] | None,
    list[pose.Result | None] | None,
    # list[face.Result | None] | None,
    list[ceiling_projection.Result | None] | None,
]


class Visualizer:
    BONE_COLOR = (255, 0, 0)
    BONE_THICKNESS = 2

    GAZE_COLOR = (255, 255, 0)
    GAZE_THICKNESS = 5

    JOINT_RADIUS = 5
    JOINT_COLOR = (0, 255, 0)

    properties: Properties
    confidence_threshold: float

    executor: ThreadPoolExecutor

    def __init__(
        self,
        executor: ThreadPoolExecutor,
        *,
        properties: Properties,
        confidence_threshold: float,
    ) -> None:
        self.properties = properties
        self.confidence_threshold = confidence_threshold

    def __draw_skeleton_and_joints(self, frame: Frame, result: pose.Result) -> Frame:
        for _, _, keypoints in result.iter():
            for i, j in YOLO_SKELETON:
                if keypoints[i, -1] < self.confidence_threshold:
                    continue

                if keypoints[j, -1] < self.confidence_threshold:
                    continue

                start = typing.cast(cv2.typing.Point, keypoints[i, :-1].astype(int))
                end = typing.cast(cv2.typing.Point, keypoints[j, :-1].astype(int))

                cv2.line(frame, start, end, self.BONE_COLOR, self.BONE_THICKNESS)

            for keypoint in keypoints:
                if keypoint[-1] < self.confidence_threshold:
                    continue

                keypoint = typing.cast(cv2.typing.Point, keypoint.astype(np.int32))

                cv2.circle(frame, keypoint[:-1], self.JOINT_RADIUS, self.JOINT_COLOR, -1)

        return frame

    def __draw_gaze_estimation(
        self, frame: Frame, result: ceiling_projection.Result
    ) -> Frame:
        starts = result.centres
        ends = starts + 100.0 * result.directions

        print('Gaze to draw:')
        print(f'{starts = }')
        print(f'{ends = }')

        start: FloatArray1
        end: FloatArray1

        for start, end in zip(starts, ends):
            cv2.line(
                frame,
                typing.cast(cv2.typing.Point, start.astype(np.int32)),
                typing.cast(cv2.typing.Point, end.astype(np.int32)),
                self.GAZE_COLOR,
                self.GAZE_THICKNESS,
            )

        return frame

    # def __draw_face_box(self, frame: Frame, result: face.Result) -> Frame:
    # 	box: IntArray1
    # 	for box in result.boxes:
    # 		x1, y1, x2, y2 = box
    # 		cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255, 0, 0))

    # 	return frame

    def __annotate_safe(
        self,
        frame: Frame,
        poses: pose.Result | None,
        gazes: ceiling_projection.Result | None,
    ) -> Frame:
        out = frame.copy()
        out.flags.writeable = True

        if poses is not None:
            out = self.__draw_skeleton_and_joints(out, poses)

        if gazes is not None:
            out = self.__draw_gaze_estimation(out, gazes)

        return out

    @autostart
    async def stream(self) -> Fiber[Input, list[Frame] | None]:
        annotated_frames: list[Frame] | None = None

        while True:
            match (yield annotated_frames):
                case list(frames), None, None, None:
                    annotated_frames = frames

                case list(frames), poses, gazes:
                    annotated_frames = list(
                        starmap(
                            self.__annotate_safe,
                            zip(
                                frames,
                                poses or repeat(None),
                                # faces or repeat(None),
                                gazes or repeat(None),
                            ),
                        )
                    )

                case _:
                    annotated_frames = None
