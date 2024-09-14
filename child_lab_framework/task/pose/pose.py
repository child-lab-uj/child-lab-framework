from functools import lru_cache
import typing
from collections.abc import Iterable
from enum import IntEnum, auto
import numpy as np
import torch
import ultralytics
from ultralytics.engine import results as yolo
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ...util import MODELS_DIR
from ...core.stream import autostart
from ...core.video import Frame
from ...core.geometry import area_broadcast
from ...core.hardware import get_best_device
from ...typing.array import FloatArray1, FloatArray2, FloatArray3, BoolArray2, IntArray2
from ...typing.stream import Fiber
from .duplication import deduplicated
from .keypoint import YoloKeypoint


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
        self.boxes, self.keypoints, self.actors = Result.__ordered(
            typing.cast(BatchedBoxes, boxes.data.cpu().numpy()),  # pyright: ignore
            typing.cast(BatchedKeypoints, keypoints.data.cpu().numpy()),  # pyright: ignore
            actors
        )

    def iter(self) -> Iterable[tuple[Actor, Box, Keypoints]]:
        return zip(self.actors, self.boxes, self.keypoints)

    @property
    @lru_cache(1)
    def centres(self) -> FloatArray2:
        keypoints = self.keypoints.view()
        return (
            keypoints[:, YoloKeypoint.RIGHT_SHOULDER, :2] +
            keypoints[:, YoloKeypoint.LEFT_SHOULDER, :2]
        ) / 2.0

    @property
    @lru_cache(1)
    def depersonificated_keypoints(self) -> FloatArray2:
        return np.concatenate(self.keypoints)


def common_points_indicator(pose1: Result, pose2: Result, probability_threshold: float) -> BoolArray2:
    joint_probabilities = pose1.keypoints.view()[:, :, 2] * pose2.keypoints.view()[:, :, 2]
    return joint_probabilities >= probability_threshold


def shoulders_depth(pose: Result, depth: FloatArray2) -> FloatArray2:
    shoulders: IntArray2 = pose.keypoints[:, [YoloKeypoint.LEFT_SHOULDER, YoloKeypoint.RIGHT_SHOULDER], :].astype(np.int32)

    left_x = shoulders.view()[:, 0, 0]
    left_y = shoulders.view()[:, 0, 1]
    right_x = shoulders.view()[:, 1, 0]
    right_y = shoulders.view()[:, 1, 1]

    return depth.view()[
        [left_y, right_y],
        [left_x, right_x]
    ]


class Estimator:
    MODEL_PATH: str = str(MODELS_DIR / 'yolov8x-pose-p6.pt')

    max_detections: int
    threshold: float  # NOTE: currently unused, but may remain for future use
    detector: ultralytics.YOLO
    executor: ThreadPoolExecutor
    device: torch.device

    def __init__(self, executor: ThreadPoolExecutor, *, max_detections: int, threshold: float) -> None:
        self.max_detections = max_detections
        self.threshold = threshold

        self.executor = executor
        self.detector = ultralytics.YOLO(
            model=self.MODEL_PATH,
            task='pose',
            verbose=False
        )

        self.device = get_best_device()

    # NOTE heuristic: Children have substantially smaller bounding boxes than adults
    def __detect_child(self, boxes: yolo.Boxes) -> int | None:
        if len(boxes) < 2:
            return None

        boxes_data = typing.cast(torch.Tensor, boxes.data)

        return int(
            torch
            .argmin(area_broadcast(boxes_data))
            .item()
        )

    # TODO: use detector.predict and self.__interpret here
    def predict(self, frame: Frame) -> Result | None:
        raise NotImplementedError()

    # TODO: JIT
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
    async def stream(self) -> Fiber[list[Frame] | None, list[Result | None] | None]:
        detector = self.detector
        executor = self.executor
        loop = asyncio.get_running_loop()
        device = self.device

        results: list[Result | None] | None = None

        while True:
            match (yield results):
                case list(frames):
                    detections = await loop.run_in_executor(
                        executor,
                        lambda: detector.predict(frames, stream=False, verbose=False, device=device)
                    )

                    results = [self.__interpret(detection) for detection in detections]

                case _:
                    results = None
