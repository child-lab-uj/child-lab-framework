import asyncio
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import starmap
import numpy as np
import typing

from . import mtcnn
from .. import pose
from ...core.sequence import imputed_with_reference_inplace
from ...typing.video import Frame
from ...typing.stream import Fiber
from ...typing.array import FloatArray1, FloatArray2


type Input = tuple[
    list[Frame] | None,
    list[pose.Result | None] | None
]


@dataclass
class Result:
    boxes: FloatArray2
    confidences: FloatArray1


class Estimator:
    threshold: float
    detector: mtcnn.Detector

    executor: ThreadPoolExecutor

    def __init__(self, executor: ThreadPoolExecutor, *, threshold: float) -> None:
        self.threshold = threshold
        self.detector = mtcnn.Detector(threshold=threshold)

    def predict(self, frame: Frame, poses: pose.Result) -> Result | None:
        detector = self.detector
        threshold = self.threshold

        face_boxes: list[FloatArray2 | None] = []
        confidences: list[list[float] | None] = []

        for box in poses.boxes:
            box_x1, box_y1, box_x2, box_y2 = box.astype(np.int32)
            actor_cropped = frame[box_y1:box_y2, box_x1:box_x2, ...]

            match detector.predict(actor_cropped):
                case mtcnn.Result(boxes, conf, _) if conf >= threshold:
                    face_box = typing.cast(FloatArray2, boxes[0, ...])
                    face_box[0] += box_y1
                    face_box[1] += box_x1
                    face_box[2] += box_y1
                    face_box[3] += box_x1

                    face_boxes.append(face_box)
                    confidences.append(conf[0])

                case _:
                    face_boxes.append(None)
                    confidences.append(None)

        return Result(
            np.stack(imputed_with_reference_inplace(face_boxes)),
            np.stack(imputed_with_reference_inplace(confidences))
        )

    def __predict_safe(self, frame: Frame, poses: pose.Result | None) -> Result | None:
        if poses is None:
            return None

        return self.predict(frame, poses)

    async def stream(self) -> Fiber[Input | None, list[Result | None] | None]:
        executor = self.executor
        loop = asyncio.get_running_loop()

        results: list[Result | None] | None = None

        while True:
            match (yield results):
                case list(frames), list(poses):
                    results = await loop.run_in_executor(
                        executor,
                        lambda: list(starmap(
                            self.__predict_safe,
                            zip(frames, poses)
                        ))
                    )

                case _:
                    results = None
