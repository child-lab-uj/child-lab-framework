import asyncio
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import (
    BaseOptions,
)
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
    VisionTaskRunningMode,
)
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarker,
    FaceLandmarkerOptions,
)

from ...core.sequence import imputed_with_zeros_reference_inplace
from ...core.stream import autostart
from ...core.video import Frame, cropped
from ...typing.array import FloatArray1, FloatArray2, FloatArray3
from ...typing.stream import Fiber
from ...util import MODELS_DIR
from .. import pose


def to_numpy(self: NormalizedLandmark) -> FloatArray2:
    return np.array(
        [self.x, self.y, self.z, self.presence, self.visibility], dtype=np.float32
    )


NormalizedLandmark.to_numpy = to_numpy  # pyright: ignore


def unpacked_and_renormalized(
    landmarks: list[NormalizedLandmark], landmarks_frame: Frame, result_frame: Frame
) -> list[FloatArray1]:
    result_height, result_width, _ = result_frame.shape
    source_height, source_width, _ = landmarks_frame.shape

    y_scale = source_height / result_height
    x_scale = source_width / result_width

    return [
        np.array(
            [
                (landmark.x or 0.0) * x_scale,
                (landmark.y or 0.0) * y_scale,
                landmark.z or 0.0,
                landmark.presence or 0.0,
                landmark.visibility or 0.0,
            ],
            dtype=np.float32,
        )
        for landmark in landmarks
    ]


class Eye(Enum):
    # right, top, left, bottom,
    Right = [469, 470, 471, 472]
    Left = [474, 475, 476, 477]


class Result:
    landmarks: FloatArray3  # [person, landmark, axis]
    transformation_matrices: FloatArray2

    def __init__(
        self, landmarks: list[FloatArray2], transformation_matrices: list[FloatArray1]
    ) -> None:
        self.landmarks = np.stack(landmarks)
        self.transformation_matrices = np.stack(transformation_matrices)

    def __repr__(self) -> str:
        landmarks = self.landmarks
        transformation_matrices = self.transformation_matrices
        return f'Result:\n{landmarks = },\n\n{transformation_matrices = }'

    def eyes(self, eye: Eye) -> FloatArray3:
        return self.landmarks[:, [469, 470, 471, 472], :]


class Estimator:
    MODEL_PATH: str = str(MODELS_DIR / 'face_landmarker.task')

    max_results: int
    detection_threshold: float
    tracking_threshold: float
    detector: FaceLandmarker
    executor: ThreadPoolExecutor

    def __init__(
        self,
        executor: ThreadPoolExecutor,
        *,
        max_results: int,
        detection_threshold: float,
        tracking_threshold: float,
    ) -> None:
        self.max_results = max_results
        self.detection_threshold = detection_threshold
        self.tracking_threshold = tracking_threshold

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.MODEL_PATH),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_faces=max_results,
            min_face_detection_confidence=detection_threshold,
            min_tracking_confidence=tracking_threshold,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )

        self.detector = FaceLandmarker.create_from_options(options)
        self.executor = executor

    def predict(self, frame: Frame, boxes: pose.BatchedBoxes) -> Result | None:
        humans = cropped(frame, boxes)

        landmarks: list[FloatArray2 | None] = []
        matrices: list[FloatArray1 | None] = []

        for human_image in humans:
            result = self.detector.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(human_image, cv2.COLOR_BGR2RGB),
                )
            )

            if len(result.face_landmarks) == 0:
                landmarks.append(None)
                matrices.append(None)
                continue

            first_detection = np.stack(
                unpacked_and_renormalized(result.face_landmarks[0], human_image, frame)
            )
            first_matrix = result.facial_transformation_matrixes[0]

            landmarks.append(first_detection)
            matrices.append(first_matrix)

        imputed_landmarks = imputed_with_zeros_reference_inplace(landmarks)
        if imputed_landmarks is None:
            return None

        imputed_matrices = imputed_with_zeros_reference_inplace(matrices)
        if imputed_matrices is None:
            return None

        return Result(imputed_landmarks, imputed_matrices)

    @autostart
    async def stream(
        self,
    ) -> Fiber[
        tuple[list[Frame] | None, list[pose.Result | None] | None],
        list[Result | None] | None,
    ]:
        executor = self.executor
        loop = asyncio.get_running_loop()

        results: list[Result | None] | None = None

        while True:
            match (yield results):
                case list(frames), list(poses):
                    results = await loop.run_in_executor(
                        executor,
                        lambda: [
                            self.predict(frame, frame_poses.boxes)
                            if frame_poses is not None
                            else None
                            for frame, frame_poses in zip(frames, poses)
                        ],
                    )

                case _:
                    results = None
