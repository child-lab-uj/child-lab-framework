import asyncio
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import starmap

import cv2
import numpy as np

from ...core.sequence import imputed_with_reference_inplace
from ...core.stream import InvalidArgumentException
from ...core.video import Properties
from ...typing.array import FloatArray1, FloatArray2, IntArray1
from ...typing.stream import Fiber
from ...typing.video import Frame
from .. import pose, visualization
from ..pose.keypoint import YoloKeypoint
from . import mtcnn

type Input = tuple[list[Frame] | None, list[pose.Result | None] | None]


@dataclass
class Result:
    boxes: FloatArray2
    confidences: FloatArray1

    def visualize(
        self,
        frame: Frame,
        frame_properties: Properties,
        configuration: visualization.Configuration,
    ) -> Frame:
        draw_boxes = configuration.face_draw_bounding_boxes
        draw_confidences = configuration.face_draw_confidence

        if draw_boxes:
            color = configuration.face_bounding_box_color
            thickness = configuration.face_bounding_box_thickness

            box: IntArray1
            for box in self.boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        if draw_confidences:
            ...  # TODO: annotate bounding boxes with confidence

        return frame


class Estimator:
    threshold: float
    detector: mtcnn.Detector

    input: Properties

    executor: ThreadPoolExecutor

    def __init__(
        self, executor: ThreadPoolExecutor, *, input: Properties, threshold: float
    ) -> None:
        self.executor = executor
        self.threshold = threshold
        self.input = input
        self.detector = mtcnn.Detector(threshold=threshold)

    def predict(self, frame: Frame, poses: pose.Result) -> Result | None:
        detector = self.detector
        threshold = self.threshold

        face_boxes: list[FloatArray1 | None] = []
        confidences: list[list[float] | None] = []

        box: FloatArray1
        keypoints: FloatArray2
        i: int
        face_box: FloatArray1
        face_box_x1: float
        face_box_y1: float
        face_box_width: float
        face_box_height: float
        face_box_x2: float
        face_box_y2: float
        nose_x: float
        nose_y: float

        for box, keypoints in zip(poses.boxes, poses.keypoints):
            box_x1, box_y1, box_x2, box_y2 = box.astype(np.int32)[:4]
            actor_cropped = frame[box_y1:box_y2, box_x1:box_x2, ...]

            match detector.predict(actor_cropped):
                case mtcnn.Result(boxes, confs, _) if np.max(confs) >= threshold:
                    for i in np.flip(np.argsort(confs)):
                        face_box = boxes[i]
                        face_box_x1, face_box_y1, face_box_width, face_box_height = (
                            face_box
                        )
                        face_box_x2 = face_box_x1 + face_box_width
                        face_box_y2 = face_box_y1 + face_box_height

                        nose_x, nose_y, *_ = keypoints[YoloKeypoint.NOSE]
                        nose_x -= box_x1
                        nose_y -= box_y1

                        if (
                            nose_x < face_box_x1
                            or face_box_x2 < nose_x
                            or nose_y < face_box_y1
                            or face_box_y2 < nose_y
                        ):
                            continue

                        break
                    else:
                        continue

                    face_box[0] += box_x1
                    face_box[1] += box_y1
                    face_box[2] = face_box_x2 + box_x1
                    face_box[3] = face_box_y2 + box_y1

                    face_boxes.append(face_box)

                    confidence = confs[i]
                    confidences.append(confidence)

                case _:
                    face_boxes.append(None)
                    confidences.append(None)

        if len(face_boxes) == 0 or len(confidences) == 0:
            return None

        match (
            imputed_with_reference_inplace(face_boxes),
            imputed_with_reference_inplace(confidences),
        ):
            case list(face_boxes_imputed), list(confidences_imputed):
                return Result(np.stack(face_boxes_imputed), np.stack(confidences_imputed))

            case _:
                return None

    def predict_batch(
        self,
        frames: list[Frame],
        poses: list[pose.Result],
    ) -> list[Result] | None:
        return imputed_with_reference_inplace(
            list(starmap(self.predict, zip(frames, poses)))
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
                        lambda: list(starmap(self.__predict_safe, zip(frames, poses))),
                    )

                case _, _:
                    results = None

                case _:
                    raise InvalidArgumentException()
