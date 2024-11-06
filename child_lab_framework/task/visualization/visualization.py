import typing
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat, starmap

import cv2
import cv2.text
import numpy as np

from ...core.video import Frame, Properties
from ...typing.array import FloatArray1, FloatArray2, IntArray1
from ...typing.stream import Fiber
from .. import face, pose, emotion
from ..gaze import ceiling_projection
from ..pose.keypoint import YOLO_SKELETON

type Input = tuple[
    list[Frame] | None,
    list[pose.Result | None] | None,
    list[ceiling_projection.Result | None] | None,
]


class Visualizer:
    BONE_COLOR = (0.0, 0.0, 255.0, 1.0)
    BONE_THICKNESS = 2

    POSE_BOUNDING_BOX_COLOR = (0.0, 255.0, 0.0, 1.0)
    POSE_BOUNDING_BOX_THICKNESS = 4

    FACE_BOUNDING_BOX_COLOR = (255.0, 0.0, 0.0, 1.0)
    FACE_BOUNDING_BOX_THICKNESS = 3

    GAZE_COLOR = (0.0, 255.0, 255.0, 1.0)
    CORRECTED_GAZE_COLOR = (255.0, 255.0, 0.0, 1.0)
    GAZE_THICKNESS = 5

    JOINT_RADIUS = 5
    JOINT_COLOR = (0.0, 255.0, 0.0, 1.0)

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
        actor_keypoints: FloatArray2

        for actor_keypoints in result.keypoints:
            for i, j in YOLO_SKELETON:
                if actor_keypoints[i, -1] < self.confidence_threshold:
                    continue

                if actor_keypoints[j, -1] < self.confidence_threshold:
                    continue

                start = typing.cast(cv2.typing.Point, actor_keypoints[i, :-1].astype(int))
                end = typing.cast(cv2.typing.Point, actor_keypoints[j, :-1].astype(int))

                cv2.line(frame, start, end, self.BONE_COLOR, self.BONE_THICKNESS)

            for keypoint in actor_keypoints:
                if keypoint[-1] < self.confidence_threshold:
                    continue

                keypoint = typing.cast(cv2.typing.Point, keypoint.astype(int))

                cv2.circle(frame, keypoint[:-1], self.JOINT_RADIUS, self.JOINT_COLOR, -1)

        return frame

    def __draw_pose_boxes(self, frame: Frame, result: pose.Result) -> Frame:
        color = self.POSE_BOUNDING_BOX_COLOR
        thickness = self.POSE_BOUNDING_BOX_THICKNESS

        box: IntArray1
        for box in result.boxes.astype(int):
            x1, y1, x2, y2, *_ = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)  # type: ignore

        return frame

    def __draw_gaze_estimation(
        self, frame: Frame, result: ceiling_projection.Result
    ) -> Frame:
        starts = result.centres
        ends = starts + 100.0 * result.directions

        start: FloatArray1
        end: FloatArray1

        thickness = self.GAZE_THICKNESS

        for start, end, was_corrected in zip(starts, ends, result.was_corrected):
            color = self.CORRECTED_GAZE_COLOR if was_corrected else self.GAZE_COLOR

            cv2.line(
                frame,
                typing.cast(cv2.typing.Point, start.astype(np.int32)),
                typing.cast(cv2.typing.Point, end.astype(np.int32)),
                color,
                thickness,
            )

        return frame

    def __draw_face_box(self, frame: Frame, result: face.Result) -> Frame:
        color = self.FACE_BOUNDING_BOX_COLOR
        thickness = self.FACE_BOUNDING_BOX_THICKNESS

        box: IntArray1
        for box in result.boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        return frame

    def __draw_emotions_text(self, frame: Frame, result: emotion.Result) -> Frame:
        color = self.FACE_BOUNDING_BOX_COLOR
        for value, box in zip(result.emotions, result.boxes):
            cv2.putText(
                frame,
                str(value),
                [box[0], box[3]],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                color,
                2,
            )
        return frame

    def __annotate_safe(
        self,
        frame: Frame,
        poses: pose.Result | None,
        faces: face.Result | None,
        gazes: ceiling_projection.Result | None,
        emotions: emotion.Result | None,
    ) -> Frame:
        out = frame.copy()
        out.flags.writeable = True

        if poses is not None:
            out = self.__draw_skeleton_and_joints(out, poses)
            out = self.__draw_pose_boxes(out, poses)

        if faces is not None:
            out = self.__draw_face_box(out, faces)

        if gazes is not None:
            out = self.__draw_gaze_estimation(out, gazes)

        if emotions is not None:
            out = self.__draw_emotions_text(out, emotions)

        return out

    def annotate_batch(
        self,
        frames: list[Frame],
        poses: list[pose.Result] | None,
        faces: list[face.Result] | None,
        gazes: list[ceiling_projection.Result] | None,
        emotions: list[emotion.Result] | None,
    ) -> list[Frame]:
        return list(
            starmap(
                self.__annotate_safe,
                zip(
                    frames,
                    poses or repeat(None),
                    faces or repeat(None),
                    gazes or repeat(None),
                    emotions or repeat(None),
                ),
            )
        )

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
