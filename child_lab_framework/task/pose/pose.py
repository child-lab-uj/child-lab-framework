
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
from ...typing.array import FloatArray1, FloatArray2, FloatArray3
from ...typing.stream import Fiber


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

    def __init__(
            self,
            n_detections: int,
            boxes: yolo.Boxes,
            keypoints: yolo.Keypoints,
            actors: list[Actor]
    ) -> None:
        self.n_detections = n_detections
        self.boxes = typing.cast(BatchedBoxes, boxes.numpy().data)
        self.keypoints = typing.cast(BatchedKeypoints, keypoints.numpy().data)
        self.actors = actors

    def iter(self) -> Iterable[tuple[Actor, Box, Keypoints]]:
        return zip(self.actors, self.boxes, self.keypoints)


def __area(rect: torch.Tensor) -> torch.Tensor:
    width = rect[3] - rect[1]
    height = rect[2] - rect[0]
    return width * height

area = typing.cast(
    Callable[[torch.Tensor], torch.Tensor],
    torch.vmap(__area, 0)
)


class Estimator:
    MODEL_PATH: str = str(MODELS_DIR / 'yolov8x-pose-p6.pt')

    max_results: int
    threshold: float  # NOTE: currently unused, but may remain for future use
    detector: ultralytics.YOLO

    def __init__(self, *, max_results: int, threshold: float) -> None:
        self.max_results = max_results
        self.threshold = threshold

        self.detector = ultralytics.YOLO(
            model=self.MODEL_PATH,
            task='pose',
            verbose=False
        )

    # heuristic: Children have substantially smaller bounding boxes than adults
    def __detect_child(self, boxes: yolo.Boxes) -> int | None:
        if len(boxes) == 1:
            return None

        boxes_data = typing.cast(torch.Tensor, boxes.data)

        return int(
            torch
            .argmin(area(boxes_data))
            .item()
        )

    def __interpret(self, detections: yolo.Results) -> Result | None:
        boxes: yolo.Boxes | None = detections.boxes
        keypoints: yolo.Keypoints | None = detections.keypoints

        if boxes is None or keypoints is None:
            return None

        n_detections = len(boxes)
        actors = [Actor.ADULT for _ in range(n_detections)]

        if i := self.__detect_child(boxes):
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
