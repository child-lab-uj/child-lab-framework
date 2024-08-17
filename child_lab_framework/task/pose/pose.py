from operator import itemgetter
import typing
from typing import Any, Callable, Iterable, Literal
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
import numpy as np
import torch
import ultralytics
from ultralytics.engine import results as yolo

from ...util import MODELS_DIR
from ...core.stream import autostart
from ...core.video import Frame
from ...core.geometry import area_broadcast
from ...typing.array import FloatArray1, FloatArray2, FloatArray3
from ...typing.stream import Fiber
from .duplication import deduplicated


type Box = FloatArray1
type BatchedBoxes = FloatArray2

type Keypoints = FloatArray2
type BatchedKeypoints = FloatArray3


class Actor(IntEnum):
    ADULT = auto()
    CHILD = auto()


class Result:
    n_detections: int
    boxes: BatchedBoxes
    keypoints: BatchedKeypoints
    actors: list[Actor]

    # NOTE heuristic: sitting actors preserve lexicographic order of bounding boxes
    @staticmethod
    def __ordered(
        boxes: BatchedBoxes,
        keypoints: BatchedKeypoints,
        actors: list[Actor]
    ) -> tuple[BatchedBoxes, BatchedKeypoints, list[Actor]]:
        order = np.lexsort(np.flipud(boxes.T))

        boxes = boxes[order]
        keypoints = keypoints[order]
        actors = [actors[i] for i in order]

        return boxes, keypoints, actors

    def __init__(
            self,
            n_detections: int,
            boxes: yolo.Boxes,
            keypoints: yolo.Keypoints,
            actors: list[Actor]
    ) -> None:
        self.n_detections = n_detections
        (self.boxes, self.keypoints, self.actors) = Result.__ordered(
            typing.cast(BatchedBoxes, boxes.numpy().data),
            typing.cast(BatchedKeypoints, keypoints.numpy().data),
            actors
        )

    def iter(self) -> Iterable[tuple[Actor, Box, Keypoints]]:
        return zip(self.actors, self.boxes, self.keypoints)


class Estimator:
    MODEL_PATH: str = str(MODELS_DIR / 'yolov8x-pose-p6.pt')

    max_detections: int
    threshold: float  # NOTE: currently unused, but may remain for future use
    detector: ultralytics.YOLO

    def __init__(self, *, max_detections: int, threshold: float) -> None:
        self.max_detections = max_detections
        self.threshold = threshold

        self.detector = ultralytics.YOLO(
            model=self.MODEL_PATH,
            task='pose',
            verbose=False
        )

    # NOTE heuristic: Children have substantially smaller bounding boxes than adults
    def __detect_child(self, boxes: yolo.Boxes) -> int | None:
        if len(boxes) == 1:
            return None

        boxes_data = typing.cast(torch.Tensor, boxes.data)

        return int(
            torch
            .argmin(area_broadcast(boxes_data))
            .item()
        )

    def __interpret(self, detections: yolo.Results) -> Result | None:
        boxes = detections.boxes
        keypoints = detections.keypoints

        if boxes is None or keypoints is None:
            return None

        boxes, keypoints = deduplicated(boxes, keypoints, self.max_detections)

        n_detections = len(boxes)
        actors = [Actor.ADULT for _ in range(n_detections)]

        if (i := self.__detect_child(boxes)) is not None:
            actors[i] = Actor.CHILD

        return Result(
            n_detections,
            boxes,
            keypoints,
            actors
        )

    @autostart
    def stream(self) -> Fiber[list[Frame] | None, list[Result | None] | None]:
        detector = self.detector
        results: list[Result | None] | None = None

        while True:
            match (yield results):
                case list(frames):
                    detections = detector.predict(frames, stream=True, verbose=False)
                    results = [self.__interpret(detection) for detection in detections]

                case _:
                    results = None
