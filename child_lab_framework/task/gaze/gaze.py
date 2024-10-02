import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import starmap

import cv2
import mini_face as mf
import numpy as np
from icecream import ic

from ...core.sequence import imputed_with_zeros_reference_inplace
from ...core.video import Frame, Properties
from ...typing.array import FloatArray3, IntArray1
from ...typing.stream import Fiber
from ...util import MODELS_DIR as MODELS_ROOT
from .. import face


@dataclass
class Result2d:
    eyes: FloatArray3
    directions: FloatArray3


@dataclass
class Result:
    eyes: FloatArray3
    directions: FloatArray3

    # A prototype, currently unused method which can be used to display gaze estimation on side views
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
    MODELS_DIR = str(MODELS_ROOT / 'mini_face')

    properties: Properties

    optical_center_x: float
    optical_center_y: float

    extractor: mf.gaze.Extractor

    executor: ThreadPoolExecutor

    def __init__(self, executor: ThreadPoolExecutor, *, properties: Properties) -> None:
        self.executor = executor

        self.properties = properties

        calibration = properties.calibration
        self.optical_center_x, self.optical_center_y = calibration.optical_center

        self.extractor = mf.gaze.Extractor(
            mode=mf.PredictionMode.VIDEO,
            focal_length=calibration.focal_length,
            optical_center=calibration.optical_center,
            models_directory=self.MODELS_DIR,
        )

    def predict(self, frame: Frame, faces: face.Result) -> Result | None:
        extractor = self.extractor

        cx = self.optical_center_x
        cy = self.optical_center_y
        center_shift = np.array((cx, cy), dtype=np.float32)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore

        eyes: list[FloatArray3 | None] = []
        directions: list[FloatArray3 | None] = []

        box: IntArray1

        for box in faces.boxes:
            region = box.astype(np.uint32)
            region[2] -= region[0]
            region[3] -= region[1]

            detection = extractor.predict(frame, region)

            if detection is None:
                eyes.append(None)
                directions.append(None)
                continue

            eyes.append(detection.eyes + center_shift)
            directions.append(detection.directions)

        match (
            imputed_with_zeros_reference_inplace(eyes),
            imputed_with_zeros_reference_inplace(directions),
        ):
            case list(eyes_imputed), list(directions_imputed):
                return Result(
                    np.concatenate(eyes_imputed, axis=0),
                    np.concatenate(directions_imputed, axis=0),
                )

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
