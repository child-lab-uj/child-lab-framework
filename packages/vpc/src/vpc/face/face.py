from dataclasses import dataclass
from typing import TypedDict

import cv2
import kornia
import torch
from jaxtyping import Float, UInt8
from video_io.annotation import Color
from video_io.frame import ArrayRgbFrame, TensorRgbFrame

from .. import pose
from ..pose.model import YoloKeypoint

__all__ = ['Estimator', 'Result', 'VisualizationContext']


class VisualizationContext(TypedDict):
    face_draw_bounding_boxes: bool
    face_bounding_box_color: Color
    face_bounding_box_thickness: int

    face_draw_confidence: bool

    face_blur: bool


@dataclass(frozen=True)
class Result:
    boxes: Float[torch.Tensor, 'n_detections 4']
    confidences: Float[torch.Tensor, ' n_detections']

    def draw(
        self,
        frame: ArrayRgbFrame,
        context: VisualizationContext,
    ) -> ArrayRgbFrame:
        if context['face_draw_bounding_boxes']:
            color = context['face_bounding_box_color']
            thickness = context['face_bounding_box_thickness']
            blur = context['face_blur']

            box: list[int]
            for box in self.boxes.cpu().to(torch.int32).tolist():
                x1, y1, x2, y2 = box

                if x2 <= x1 or y2 <= y1:
                    continue

                if blur:
                    frame_view = frame[y1:y2, x1:x2, :]
                    blurred = cv2.blur(frame_view, (59, 59))
                    frame[y1:y2, x1:x2, :] = blurred

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # TODO: annotate the bounding boxes with confidence.
        if context['face_draw_confidence']:
            pass

        return frame


class Estimator:
    detector: kornia.contrib.FaceDetector
    device: torch.device

    def __init__(
        self,
        confidence_threshold: float,
        suppression_threshold: float,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device('cpu')
        self.detector = kornia.contrib.FaceDetector(
            confidence_threshold=confidence_threshold,
            nms_threshold=suppression_threshold,
        ).to(self.device)

    @torch.inference_mode()
    def predict(self, frame: TensorRgbFrame, poses: pose.Result) -> Result | None:
        detections = self.detector.forward(
            torch.tensor(frame).to(self.device, torch.float32).unsqueeze(0)
        )

        if len(detections) == 0:
            return None

        detection = detections[0]  # There will always be one result for one frame

        return self.__match_faces_with_actors(detection, poses)

    @torch.inference_mode()
    def predict_batch(
        self,
        frames: UInt8[torch.Tensor, 'batch 3 height width'],
        poses: list[pose.Result],
    ) -> list[Result | None]:
        faces = self.detector.forward(frames.to(self.device, torch.float32))
        return [
            self.__match_faces_with_actors(frame_faces, frame_poses)
            for frame_faces, frame_poses in zip(faces, poses)
        ]

    def __match_faces_with_actors(
        self,
        faces: Float[torch.Tensor, 'n_detections 15'],
        poses: pose.Result,
    ) -> Result | None:
        device = self.device

        descending_confidence = faces[:, 14].argsort(descending=True)
        faces = faces[descending_confidence, :]

        n_actors = len(poses.boxes)
        n_faces = faces.shape[0]

        if n_faces == 0:
            return None

        noses = (
            poses.keypoints[:, YoloKeypoint.NOSE, :2]
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

        result_boxes = boxes_and_confidences[..., :4]
        result_confidences = boxes_and_confidences[..., 4]

        return Result(result_boxes, result_confidences)
