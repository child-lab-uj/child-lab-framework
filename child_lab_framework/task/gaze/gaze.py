import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import repeat, starmap

import mini_face as mf
import numpy as np

from ...core.sequence import (
    imputed_with_reference_inplace,
    imputed_with_zeros_reference_inplace,
)
from ...core.stream import InvalidArgumentException
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
    was_projected: bool = field(default=False)


type Input = tuple[list[Frame], list[face.Result | None] | None]


class Estimator:
    MODELS_DIR = MODELS_ROOT / 'model'

    input: Properties

    optical_center_x: float
    optical_center_y: float

    extractor: mf.gaze.Extractor

    executor: ThreadPoolExecutor

    def __init__(
        self,
        executor: ThreadPoolExecutor,
        *,
        input: Properties,
        wild: bool = False,
        multiple_views: bool = False,
        limit_angles: bool = False,
    ) -> None:
        self.executor = executor

        self.input = input

        calibration = input.calibration
        self.optical_center_x, self.optical_center_y = calibration.optical_center

        self.extractor = mf.gaze.Extractor(
            mode=mf.PredictionMode.VIDEO,
            focal_length=calibration.focal_length,
            optical_center=calibration.optical_center,
            fps=input.fps,
            models_directory=self.MODELS_DIR,
        )

    def predict(self, frame: Frame, faces: face.Result) -> Result | None:
        extractor = self.extractor

        cx = self.optical_center_x
        cy = self.optical_center_y
        center_shift = np.array((cx, cy, 0.0), dtype=np.float32).reshape(1, 1, -1)

        eyes: list[FloatArray3 | None] = []
        directions: list[FloatArray3 | None] = []  # in fact arrays are 1 x 2 x 3

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
            directions.append(detection.directions)  # type: ignore

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

    def predict_batch(
        self,
        frames: list[Frame],
        faces: list[face.Result],
    ) -> list[Result] | None:
        return imputed_with_reference_inplace(
            list(starmap(self.predict, zip(frames, faces)))
        )

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
                case list(frames), faces:
                    results = await loop.run_in_executor(
                        executor,
                        lambda: list(
                            starmap(
                                self.__predict_safe, zip(frames, faces or repeat(None))
                            )
                        ),
                    )

                case None:
                    results = None

                case _:
                    raise InvalidArgumentException()
