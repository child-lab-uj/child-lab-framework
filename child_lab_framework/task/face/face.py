from collections.abc import Generator
from dataclasses import dataclass
from typing import Self

import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision import RunningMode

from mediapipe.tasks.python import (
    BaseOptions,
)

from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarksConnections,
    FaceLandmarkerResult
)

from ...util import MODELS_DIR
from .. import pose
from ..pose import Box
from ...core.video import Frame
from ...core.stream import autostart
from ...typing.stream import Fiber
from ...typing.array import FloatArray2, FloatArray3


type Input = tuple[list[Frame], list[pose.Result]]


def to_numpy(self: NormalizedLandmark) -> FloatArray2:
    return np.array([
        self.x,
        self.y,
        self.z,
        self.presence,
        self.visibility
    ], dtype=np.float32)

NormalizedLandmark.to_numpy = to_numpy  # pyright: ignore


@dataclass(repr=False)
class Result:
    landmarks: FloatArray3  # [person, landmark, axis]
    transformation_matrices: FloatArray2

    def __repr__(self) -> str:
        landmarks = self.landmarks
        transformation_matrices = self.transformation_matrices
        return f'Result:\n{landmarks = },\n\n{transformation_matrices = }'


def __crop(frame: Frame, box: Box) -> Frame:
    start_x, start_y, end_x, end_y = box.astype(int)
    return frame[start_y:end_y, start_x:end_x]


class Estimator:
    MODEL_PATH: str = str(MODELS_DIR / 'face_landmarker.task')

    max_results: int
    detection_threshold: float
    tracking_threshold: float
    detector: FaceLandmarker

    def __init__(self, *, max_results: int, detection_threshold: float, tracking_threshold: float) -> None:
        self.max_results = max_results
        self.detection_threshold = detection_threshold
        self.tracking_treshold = tracking_threshold

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.MODEL_PATH),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=max_results,
            min_face_detection_confidence=detection_threshold,
            min_tracking_confidence=tracking_threshold,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )

        self.detector = FaceLandmarker.create_from_options(options)

    def __crop(self, frame: Frame, detction: pose.Result) -> list[Frame]:
        return [
            __crop(frame, box)
            for box in detction.boxes
        ]

    def __simple_detect(self, frame: Frame) -> Result | None:
        results = self.detector.detect(mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ))

        if len(results.face_landmarks) == 0:
            return None

        landmarks: FloatArray3 = np.stack([
            np.stack([landmark.to_numpy() for landmark in person_landmarks])
            for person_landmarks in results.face_landmarks
        ])

        matrices: FloatArray2 = np.stack(results.facial_transformation_matrixes)

        return Result(landmarks, matrices)

    # TODO: crop frames using Result
    @autostart
    def stream(self) -> Fiber[list[Frame] | None, list[Result | None] | None]:
        results: list[Result | None] | None = None

        while True:
            match (yield results):
                case list(frames):
                    results = [
                        self.__simple_detect(frame)
                        for frame in frames
                    ]

                case _:
                    results = None
