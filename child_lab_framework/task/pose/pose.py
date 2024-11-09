import asyncio
import typing
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from enum import IntEnum, auto

import cv2
import numpy as np
import torch
import ultralytics
from torchvision.transforms import Compose, Resize
from torchvision.transforms.transforms import InterpolationMode
from ultralytics.engine import results as yolo

from ...core.geometry import area_broadcast
from ...core.sequence import imputed_with_reference_inplace
from ...core.video import Frame, Properties
from ...typing.array import FloatArray1, FloatArray2, FloatArray3, IntArray1
from ...typing.stream import Fiber
from ...util import MODELS_DIR
from .duplication import deduplicated
from .keypoint import YOLO_SKELETON, YoloKeypoint

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

    __centres: FloatArray2 | None = None
    __depersonificated_keypoints: FloatArray2 | None = None

    # NOTE heuristic: sitting actors preserve lexicographic order of bounding boxes
    @staticmethod
    def __ordered(
        boxes: BatchedBoxes, keypoints: BatchedKeypoints, actors: list[Actor]
    ) -> tuple[BatchedBoxes, BatchedKeypoints, list[Actor]]:
        order = np.lexsort(np.flipud(boxes.T))

        boxes = boxes[order]
        keypoints = keypoints[order]
        actors = [actors[i] for i in order]

        return boxes, keypoints, actors

    def __init__(
        self,
        n_detections: int,
        boxes: BatchedBoxes,
        keypoints: BatchedKeypoints,
        actors: list[Actor],
    ) -> None:
        self.n_detections = n_detections
        self.boxes, self.keypoints, self.actors = Result.__ordered(
            boxes,
            keypoints,
            actors,
        )

    def iter(self) -> Iterable[tuple[Actor, Box, Keypoints]]:
        return zip(self.actors, self.boxes, self.keypoints)

    @property
    def centres(self) -> FloatArray2:
        if self.__centres is not None:
            return self.__centres

        keypoints = self.keypoints.view()

        centres = (
            keypoints[:, YoloKeypoint.RIGHT_SHOULDER, :2]
            + keypoints[:, YoloKeypoint.LEFT_SHOULDER, :2]
        ) / 2.0

        self.__centres = centres

        return centres

    @property
    def depersonificated_keypoints(self) -> FloatArray2:
        if self.__depersonificated_keypoints is not None:
            return self.__depersonificated_keypoints

        stacked_keypoints = np.concatenate(self.keypoints)
        self.__depersonificated_keypoints = stacked_keypoints

        return stacked_keypoints

    def visualize(
        self,
        frame: Frame,
        frame_properties: Properties,
        configuration: typing.Any,  # TODO: Add hint
    ) -> Frame:
        actor_keypoints: FloatArray2

        draw_skeletons = configuration.pose_draw_skeletons
        draw_boxes = configuration.pose_draw_boxes

        # TODO: draw keypoint confidence
        if draw_skeletons:
            bone_color = configuration.pose_bone_color
            bone_thickness = configuration.pose_bone_thickness
            keypoint_color = configuration.pose_keypoint_color
            keypoint_radius = configuration.pose_keypoint_radius
            keypoint_threshold = configuration.pose_keypoint_confidence_threshold

            for actor_keypoints in self.keypoints:
                for i, j in YOLO_SKELETON:
                    if actor_keypoints[i, -1] < keypoint_threshold:
                        continue

                    if actor_keypoints[j, -1] < keypoint_threshold:
                        continue

                    start = typing.cast(
                        cv2.typing.Point, actor_keypoints[i, :-1].astype(int)
                    )
                    end = typing.cast(
                        cv2.typing.Point, actor_keypoints[j, :-1].astype(int)
                    )

                    cv2.line(frame, start, end, bone_color, bone_thickness)

                for keypoint in actor_keypoints:
                    if keypoint[-1] < keypoint_threshold:
                        continue

                    keypoint = typing.cast(cv2.typing.Point, keypoint.astype(int))

                    cv2.circle(frame, keypoint[:-1], keypoint_radius, keypoint_color, -1)

        # TODO: draw bounding box confidence
        if draw_boxes:
            color = configuration.pose_bounding_box_color
            thickness = configuration.pose_bounding_box_thickness
            threshold = configuration.pose_bounding_box_confidence_threshold

            box: IntArray1
            for box in self.boxes.astype(int):
                x1, y1, x2, y2, confidence, *_ = box

                if confidence < threshold:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)  # type: ignore

        return frame


class Estimator:
    MODEL_PATH: str = str(MODELS_DIR / 'yolov11x-pose.pt')

    executor: ThreadPoolExecutor
    device: torch.device

    model: ultralytics.YOLO
    to_model: Compose

    input: Properties

    max_detections: int
    threshold: float  # NOTE: currently unused, but may remain for future use

    def __init__(
        self,
        executor: ThreadPoolExecutor,
        device: torch.device,
        *,
        input: Properties,
        max_detections: int,
        threshold: float,
    ) -> None:
        self.executor = executor
        self.device = device

        self.input = input

        self.max_detections = max_detections
        self.threshold = threshold

        self.model = ultralytics.YOLO(model=self.MODEL_PATH, task='pose', verbose=False)
        self.to_model = Compose(
            [
                Resize((640, 640), interpolation=InterpolationMode.NEAREST),
            ]
        )

    # NOTE heuristic: Children have substantially smaller bounding boxes than adults
    def __detect_child(self, boxes: yolo.Boxes) -> int | None:
        if len(boxes) < 2:
            return None

        boxes_data = typing.cast(torch.Tensor, boxes.data)

        return int(torch.argmin(area_broadcast(boxes_data)).item())

    def predict(self, frame: Frame) -> Result | None:
        result = self.model.predict(frame, verbose=False, device=self.device)

        if result is None or len(result) == 0 or result.nelements() == 0:  # type: ignore
            return None

        return self.__interpret(result[0])

    def predict_batch(self, frames: list[Frame]) -> list[Result] | None:
        device = self.device

        tensor_frames = [
            torch.permute(
                torch.from_numpy(frame).to(device, torch.float32, copy=True),
                (2, 0, 1),
            )
            for frame in frames
        ]

        frame_batch = self.to_model(torch.stack(tensor_frames)) / 255.0

        detections = self.model.predict(
            frame_batch, stream=False, verbose=False, device=device
        )

        return imputed_with_reference_inplace(
            [self.__interpret(detection) for detection in detections]
        )

    # TODO: JIT
    def __interpret(self, detections: yolo.Results) -> Result | None:
        boxes = detections.boxes
        keypoints = detections.keypoints

        if boxes is None or keypoints is None:
            return None

        if boxes.data.nelement() == 0 or keypoints.data.nelement() == 0:  # type: ignore # obvious that these are tensors :v
            return None

        boxes, keypoints = deduplicated(boxes, keypoints, self.max_detections)

        n_detections = len(boxes)
        actors = [Actor.ADULT for _ in range(n_detections)]

        if (i := self.__detect_child(boxes)) is not None:
            actors[i] = Actor.CHILD

        boxes_cpu = typing.cast(BatchedBoxes, boxes.data.cpu().numpy())  # type: ignore
        keypoints_cpu = typing.cast(BatchedKeypoints, keypoints.data.cpu().numpy())  # type: ignore

        height_rescale = float(self.input.height) / 640.0
        width_rescale = float(self.input.width) / 640.0

        boxes_cpu[..., [0, 2]] *= width_rescale
        boxes_cpu[..., [1, 3]] *= height_rescale

        keypoints_cpu[..., 0] *= width_rescale
        keypoints_cpu[..., 1] *= height_rescale

        del boxes, keypoints

        return Result(n_detections, boxes_cpu, keypoints_cpu, actors)

    async def stream(self) -> Fiber[list[Frame] | None, list[Result] | None]:
        executor = self.executor
        loop = asyncio.get_running_loop()
        device = self.device

        model = self.model
        to_model = self.to_model

        results: list[Result] | None = None

        while True:
            match (yield results):
                case list(frames):
                    tensor_frames = [
                        torch.permute(
                            torch.from_numpy(frame).to(device, torch.float32, copy=True),
                            (2, 0, 1),
                        )
                        for frame in frames
                    ]

                    frame_batch = to_model(torch.stack(tensor_frames)) / 255.0

                    detections = await loop.run_in_executor(
                        executor,
                        lambda: model.predict(
                            frame_batch, stream=False, verbose=False, device=device
                        ),
                    )

                    results = imputed_with_reference_inplace(
                        [self.__interpret(detection) for detection in detections]
                    )

                case _:
                    results = None
