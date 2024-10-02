import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import starmap

import cv2

# import mini_face as mf  # type: ignore
import numpy as np
from icecream import ic

from ....core.sequence import imputed_with_zeros_reference_inplace
from ....core.video import Frame, Properties
from ....typing.array import FloatArray2, FloatArray3, IntArray1
from ....typing.stream import Fiber
from ....util import DEV_DIR
from ... import face
from .typing import GazeExtractor

OPENFACE_DIR = DEV_DIR / 'gaze-tracking'
sys.path.append(str(OPENFACE_DIR / 'build'))
import GazeTracking as gt  # type: ignore  # noqa: E402


@dataclass
class Result2d:
    eyes: FloatArray3
    directions: FloatArray3


@dataclass
class Result:
    eyes: FloatArray3
    directions: FloatArray3

    def projected(self, fx: float, fy: float, cx: float, cy: float) -> Result2d:
        n_people = len(self.eyes)

        eyes_projection: FloatArray3 = np.zeros((n_people, 2, 2), dtype=np.float32)
        directions_projection: FloatArray3 = np.zeros((n_people, 2, 2), dtype=np.float32)

        eyes = self.eyes.view()
        directions = self.directions.view()

        eyes_z_coordinates = eyes[..., -1]

        ic(eyes_z_coordinates)

        eyes_projection[:, :, 0] = eyes[:, :, 0] * (fx / eyes_z_coordinates) + cx
        eyes_projection[:, :, 1] = eyes[:, :, 1] * (fy / eyes_z_coordinates) + cy

        directions_z_coordinates = directions[..., -1] * 1000

        ic(directions_z_coordinates)

        directions_projection[:, :, 0] = (
            directions[:, :, 0] * (fx / directions_z_coordinates) + cx
        )
        directions_projection[:, :, 1] = (
            directions[:, :, 1] * (fy / directions_z_coordinates) + cy
        )

        return Result2d(eyes_projection, directions_projection)


type Input = tuple[list[Frame], list[face.Result | None] | None]


class Estimator:
    properties: Properties

    optical_center_x: float
    optical_center_y: float

    extractor: GazeExtractor

    # NOTE: time shared accross streams - a workaround; TODO: delete ASAP
    time: float
    time_quant: float

    executor: ThreadPoolExecutor

    def __init__(self, executor: ThreadPoolExecutor, *, properties: Properties) -> None:
        self.executor = executor

        self.properties = properties
        self.optical_center_x, self.optical_center_y = (
            properties.calibration.optical_center
        )

        self.time = 0.0
        self.time_quant = 1.0 / properties.fps

        extractor: GazeExtractor = gt.GazeExtractor()
        extractor.set_camera_calibration(*properties.calibration.flat())
        self.extractor = extractor

    def predict(self, frame: Frame, faces: face.Result) -> Result | None:
        extractor = self.extractor
        timestamp = self.time
        cx = self.optical_center_x
        cy = self.optical_center_y

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore

        eyes: list[FloatArray2 | None] = []
        directions: list[FloatArray2 | None] = []

        box: IntArray1
        x1: int
        x2: int
        y1: int
        y2: int

        for box in faces.boxes:
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1

            detection = extractor.detect_gaze(frame, timestamp, (x1, y1, width, height))

            if detection is None:
                eyes.append(None)
                directions.append(None)
                continue

            eye1 = detection.eye1
            eye2 = detection.eye2

            eyes.append(
                np.array(
                    (
                        (eye1[0] + cx, eye1[1] + cy, eye1[2]),
                        (eye2[0] + cx, eye2[1] + cy, eye2[2]),
                    ),
                    dtype=np.float32,
                )
            )

            directions.append(
                np.array((detection.direction1, detection.direction2), dtype=np.float32)
            )

        self.time += self.time_quant

        match (
            imputed_with_zeros_reference_inplace(eyes),
            imputed_with_zeros_reference_inplace(directions),
        ):
            case list(eyes_imputed), list(directions_imputed):
                return Result(np.stack(eyes_imputed), np.stack(directions_imputed))

            case _:
                return None

    def __predict_safe(self, frame: Frame, faces: face.Result | None) -> Result | None:
        if faces is None:
            return None

        return self.predict(frame, faces)

    async def stream(self) -> Fiber[Input | None, list[Result | None] | None]:
        executor = self.executor
        loop = asyncio.get_running_loop()

        results: list[Result | None] | None = None

        while True:
            match (yield results):
                case list(frames), list(faces):
                    results = await loop.run_in_executor(
                        executor,
                        lambda: list(starmap(self.__predict_safe, zip(frames, faces))),
                    )

                case _:
                    results = None
