import asyncio

from itertools import starmap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import sys
import typing
import numpy as np

from .typing import GazeExtractor
from ... import face
from ....core.video import Properties, Frame
from ....core.sequence import imputed_with_reference_inplace
from ....util import DEV_DIR
from ....typing.stream import Fiber
from ....typing.array import FloatArray2, FloatArray3, IntArray1

OPENFACE_DIR = DEV_DIR / 'gaze-tracking'
sys.path.append(str(OPENFACE_DIR / 'build'))
import GazeTracking as gt  # pyright: ignore


@dataclass
class Result:
    eyes: FloatArray3
    directions: FloatArray3


type Input = tuple[
    list[Frame],
    list[face.Result | None] | None
]

class Estimator:
    properties: Properties
    extractor: GazeExtractor

    # NOTE: time shared accross streams - a workaround; TODO: delete ASAP
    time: float
    time_quant: float

    executor: ThreadPoolExecutor

    def __init__(self, executor: ThreadPoolExecutor, *, properties: Properties) -> None:
        self.executor = executor
        self.properties = properties
        self.time = 0.0
        self.time_quant = 1.0 / properties.fps

        extractor = typing.cast(
            GazeExtractor,
            gt.GazeExtractor()
        )
        extractor.set_camera_calibration(*properties.calibration.flat())

        self.extractor = extractor

    def predict(self, frame: Frame, faces: face.Result) -> Result | None:
        extractor = self.extractor
        timestamp = self.time

        eyes: list[FloatArray2 | None] = []
        directions: list[FloatArray2 | None] = []

        box: IntArray1

        for box in faces.boxes:
            detection = extractor.detect_faces(
                frame,
                timestamp,
                tuple(box)
            )

            if detection is None:
                eyes.append(None)
                directions.append(None)
                continue

            eyes.append(np.array((
                detection.eye1,
                detection.eye2
            ), dtype=np.float32))

            directions.append(np.array((
                detection.direction1,
                detection.direction2
            ), dtype=np.float32))

        self.time += self.time_quant

        return Result(
            np.stack(imputed_with_reference_inplace(eyes)),
            np.stack(imputed_with_reference_inplace(directions))
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
                case list(frames), list(faces):
                    results = await loop.run_in_executor(
                        executor,
                        lambda: list(starmap(
                            self.__predict_safe,
                            zip(frames, faces)
                        ))
                    )

                case _:
                    results = None
