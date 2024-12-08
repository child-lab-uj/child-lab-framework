import asyncio
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import starmap

import cv2
import kornia
import numpy as np
import torch

from ...core.stream import InvalidArgumentException
from ...core.video import Properties
from ...typing.array import FloatArray1, FloatArray2, IntArray1
from ...typing.stream import Fiber
from ...typing.video import Frame
from .. import pose, visualization
from ..pose.keypoint import YoloKeypoint

type Input = tuple[list[Frame] | None, list[pose.Result | None] | None]


@dataclass(frozen=True)
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
            for box in self.boxes.astype(np.int32):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        if draw_confidences:
            ...  # TODO: annotate bounding boxes with confidence

        return frame


class Estimator:
    executor: ThreadPoolExecutor | None
    device: torch.device

    input: Properties

    detector: kornia.contrib.FaceDetector

    def __init__(
        self,
        device: torch.device,
        *,
        input: Properties,
        confidence_threshold: float,
        suppression_threshold: float,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.executor = executor
        self.device = device

        self.input = input

        self.detector = kornia.contrib.FaceDetector(
            confidence_threshold=confidence_threshold,
            nms_threshold=suppression_threshold,
        ).to(device)

    @torch.no_grad
    def predict(self, frame: Frame, poses: pose.Result) -> Result | None:
        detections = self.detector.forward(
            torch.tensor(frame)
            .to(self.device, torch.float32)
            .permute(0, 2, 1)
            .unsqueeze(0)
        )

        if len(detections) == 0:
            return None

        detection = detections[0]  # There will always be one result for one frame

        return self.__match_faces_with_actors(detection, poses)

    @torch.no_grad
    def predict_batch(
        self,
        frames: list[Frame],
        poses: list[pose.Result],
    ) -> list[Result | None]:
        frame_batch = (
            torch.stack([torch.tensor(frame) for frame in frames])
            .permute(0, 3, 1, 2)
            .to(self.device, torch.float32)
        )

        predictions = self.detector.forward(frame_batch)

        return list(starmap(self.__match_faces_with_actors, zip(predictions, poses)))

    def __predict_safe(self, frame: Frame, poses: pose.Result | None) -> Result | None:
        if poses is None:
            return None

        return self.predict(frame, poses)

    def __match_faces_with_actors(
        self,
        faces: torch.Tensor,
        poses: pose.Result,
    ) -> Result | None:
        device = self.device

        descending_confidence = faces[:, 14].argsort(descending=True)
        faces = faces[descending_confidence, :]

        n_actors = poses.n_detections
        n_faces = faces.shape[0]

        if n_faces == 0:
            return None

        noses = (
            torch.tensor(poses.keypoints[:, YoloKeypoint.NOSE, :2])
            .permute(1, 0)
            .unsqueeze(-1)
            .expand(2, n_actors, n_faces)
            .to(device)
        )

        nose_x, nose_y = noses.unbind(0)

        x_lower = faces[..., 0] <= nose_x
        y_lower = faces[..., 1] <= nose_y
        x_upper = faces[..., 2] >= nose_x
        y_upper = faces[..., 3] >= nose_y

        mask = x_lower & x_upper & y_lower & y_upper

        success, where = mask.max(1)

        if not success.any():
            return None

        actor_indices = torch.arange(n_actors, dtype=torch.int32).to(device)
        face_indices = torch.arange(n_faces, dtype=torch.int32).to(device)

        which_actors = actor_indices[success]
        which_faces = face_indices[where[success]]

        boxes_and_confidences = torch.zeros((n_actors, 5), dtype=torch.float32).to(device)

        boxes_and_confidences[which_actors, :4] = faces[which_faces, :4]
        boxes_and_confidences[which_actors, 4] = faces[which_faces, 14]

        boxes_and_confidences = boxes_and_confidences.cpu().numpy()

        result_boxes = boxes_and_confidences[..., :4]
        result_confidences = boxes_and_confidences[..., 4]

        return Result(result_boxes, result_confidences)

    async def stream(self) -> Fiber[Input | None, list[Result | None] | None]:
        executor = self.executor
        if executor is None:
            raise RuntimeError(
                'Processing in the stream mode requires the Estimator to have an executor. Please pass an "executor" argument to the estimator constructor'
            )

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
