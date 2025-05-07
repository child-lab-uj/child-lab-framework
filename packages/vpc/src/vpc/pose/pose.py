from dataclasses import dataclass
from pathlib import Path
from typing import Self, TypedDict, cast

import numpy
import torch
from jaxtyping import Float, UInt8
from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    InterpolationMode,
    Resize,
)
from ultralytics import YOLO as Yolo
from ultralytics.engine.results import Results as YoloResults
from video_io.annotation import Color
from video_io.calibration import Calibration
from video_io.frame import ArrayRgbFrame, TensorRgbFrame

from .deduplication import deduplicated
from .drawing import draw_bounding_boxes, draw_keypoints

__all__ = ['Estimator', 'Result', 'Result3d']


class VisualizationContext(TypedDict):
    pose_draw_bounding_boxes: bool
    pose_bounding_box_color: Color
    pose_bounding_box_thickness: int
    pose_bounding_box_min_confidence: float

    pose_draw_keypoints: bool
    pose_bone_color: Color
    pose_bone_thickness: int
    pose_keypoint_color: Color
    pose_keypoint_radius: int
    pose_keypoint_min_confidence: float


@dataclass(slots=True)
class Result:
    boxes: Float[torch.Tensor, 'n_detections 5']
    keypoints: Float[torch.Tensor, 'n_detections 17 3']

    device: torch.device
    """
    # Invariant
    `self.boxes` and `self.components` should be located on `self.device`
    """

    @classmethod
    def ordered(
        cls,
        boxes: Float[torch.Tensor, 'n_detections 5'],
        keypoints: Float[torch.Tensor, 'n_detections 17 3'],
    ) -> Self:
        assert boxes.device == keypoints.device
        device = boxes.device

        # TODO: Implement lexicographic sorting in Torch.
        order_array = numpy.lexsort(boxes.cpu().numpy().T)
        order = torch.from_numpy(order_array).to(device)

        return cls(boxes[order], keypoints[order], device)

    def to(self, device: torch.device) -> 'Result':
        return Result(self.boxes.to(device), self.keypoints.to(device), device)

    def cpu(self) -> 'Result':
        return Result(self.boxes.cpu(), self.keypoints.cpu(), torch.device('cpu'))

    def unproject(
        self,
        calibration: Calibration,
        depth: Float[torch.Tensor, 'height width'],
    ) -> 'Result3d':
        device = self.device
        if depth.device != device:
            depth = depth.to(device)

        height, width = depth.shape
        n_actors, n_keypoints, _ = self.keypoints.shape

        flat_keypoints_with_confidence = self.keypoints.flatten(0, 1)
        keypoints = flat_keypoints_with_confidence[:, :2].clone()
        confidence = flat_keypoints_with_confidence[:, 2].unsqueeze(-1).clone()

        keypoints_truncated = keypoints.to(torch.int32)

        keypoints_depth = depth[
            keypoints_truncated[:, 1].clamp(0, height - 1),
            keypoints_truncated[:, 0].clamp(0, width - 1),
        ].reshape(-1, 1)

        cx, cy = calibration.optical_center
        fx, fy = calibration.focal_length

        keypoints[:, 0] -= cx
        keypoints[:, 1] -= cy
        keypoints *= keypoints_depth
        keypoints[:, 0] /= fx
        keypoints[:, 1] /= fy

        unprojected_keypoints = torch.cat(
            (keypoints, keypoints_depth, confidence),
            dim=1,
        ).reshape(n_actors, n_keypoints, -1)

        # TODO(#62): Unproject the bounding boxes properly.
        unprojected_boxes = self.boxes

        return Result3d(
            unprojected_boxes,
            unprojected_keypoints,
            self.device,
        )

    def draw(self, frame: ArrayRgbFrame, context: VisualizationContext) -> ArrayRgbFrame:
        if context['pose_draw_bounding_boxes']:
            draw_keypoints(
                frame,
                self.keypoints.cpu().numpy(),
                context['pose_bone_color'],
                context['pose_bone_thickness'],
                context['pose_keypoint_color'],
                context['pose_keypoint_radius'],
                context['pose_keypoint_min_confidence'],
            )

        if context['pose_draw_bounding_boxes']:
            draw_bounding_boxes(
                frame,
                self.boxes.cpu().numpy(),
                context['pose_bounding_box_color'],
                context['pose_bounding_box_thickness'],
                context['pose_bounding_box_min_confidence'],
            )

        return frame


@dataclass(slots=True)
class Result3d:
    boxes: Float[torch.Tensor, 'n_detections 6']
    keypoints: Float[torch.Tensor, 'n_detections 17 4']

    device: torch.device
    """
    # Invariant:
    `self.boxes` and `self.components` should be located on `self.device`
    """

    def to(self, device: torch.device) -> 'Result3d':
        return Result3d(self.boxes.to(device), self.keypoints.to(device), device)

    def cpu(self) -> 'Result3d':
        return Result3d(self.boxes.cpu(), self.keypoints.cpu(), torch.device('cpu'))

    def project(self, calibration: Calibration) -> Result:
        cx, cy = calibration.optical_center
        fx, fy = calibration.focal_length

        projected_keypoints = self.keypoints.clone()
        projected_keypoints[..., 0] *= fx
        projected_keypoints[..., 1] *= fy
        projected_keypoints[..., :2] /= projected_keypoints[..., 2].unsqueeze(-1)
        projected_keypoints[..., 0] += cx
        projected_keypoints[..., 1] += cy
        projected_keypoints = projected_keypoints[..., [0, 1, 3]]

        # TODO(#62): project the bounding boxes properly.
        projected_boxes = self.boxes

        return Result(projected_boxes, projected_keypoints, self.device)

    # TODO(#63): Check for order invariant after transforming and reorder the results if needed.
    def transform(self, transformation: Float[torch.Tensor, '4 4']) -> 'Result3d':
        keypoints = self.keypoints.clone()
        transformed_keypoints = torch.einsum('ij,mnj->mni', transformation, keypoints)

        # TODO(#62): transform the bounding boxes properly.
        transformed_boxes = self.boxes.clone()

        return Result3d(transformed_boxes, transformed_keypoints, self.device)


class Estimator:
    yolo: Yolo
    to_model: Compose

    max_detections: int

    device: torch.device

    def __init__(
        self,
        max_detections: int,
        yolo_checkpoint: Path,
        device: torch.device | None = None,
    ) -> None:
        self.yolo = Yolo(yolo_checkpoint, task='pose')
        self.to_model = self.to_model = Compose(  # type: ignore[no-untyped-call]
            [
                Resize(
                    (640, 640),
                    interpolation=InterpolationMode.NEAREST,
                ),  # type: ignore[no-untyped-call]
                ConvertImageDtype(torch.float32),
            ]
        )

        self.max_detections = max_detections

        self.device = device or torch.device('cpu')

    def to(self, device: torch.device) -> Self:
        self.device = device
        return self

    @torch.inference_mode()
    def predict(self, frame: TensorRgbFrame) -> Result | None:
        prediction = self.yolo.predict(
            self.to_model(frame).to(self.device),
            stream=False,
            verbose=False,
            device=self.device,
        )

        if prediction is None or len(prediction) == 0 or prediction.nelements() == 0:  # type: ignore
            return None

        _, height, width = frame.shape
        return self.__interpret(prediction[0], height, width)

    @torch.inference_mode()
    def predict_batch(
        self,
        frames: UInt8[torch.Tensor, 'batch 3 height width'],
    ) -> list[Result | None]:
        detections = self.yolo.predict(
            self.to_model(frames).to(self.device),
            stream=False,
            verbose=False,
            device=self.device,
        )

        *_, height, width = frames.shape
        return [self.__interpret(detection, height, width) for detection in detections]

    def __interpret(
        self,
        detections: YoloResults,
        original_height: int,
        original_width: int,
    ) -> Result | None:
        boxes = detections.boxes
        keypoints = detections.keypoints

        if boxes is None or keypoints is None:
            return None

        boxes_data: torch.Tensor = cast(torch.Tensor, boxes.data)
        keypoints_data: torch.Tensor = cast(torch.Tensor, keypoints.data)

        if boxes_data.nelement() == 0 or keypoints_data.nelement() == 0:
            return None

        boxes, keypoints = deduplicated(boxes, keypoints, self.max_detections)
        boxes_data = cast(torch.Tensor, boxes.data)
        keypoints_data = cast(torch.Tensor, keypoints.data)

        height_rescale = original_height / 640.0
        width_rescale = original_width / 640.0

        boxes_data[..., [0, 2]] *= width_rescale
        boxes_data[..., [1, 3]] *= height_rescale

        keypoints_data[..., 0] *= width_rescale
        keypoints_data[..., 1] *= height_rescale

        return Result.ordered(boxes_data, keypoints_data)
