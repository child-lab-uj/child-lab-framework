from collections.abc import Sequence
from dataclasses import dataclass
from itertools import starmap
from pathlib import Path
from typing import ClassVar, Literal, TypedDict, cast

import cv2 as opencv
import mini_face
import numpy
import torch
from jaxtyping import Float, Int, UInt8, UInt32
from more_itertools import first_true
from video_io.annotation import Color
from video_io.calibration import Calibration
from video_io.frame import ArrayRgbFrame, TensorRgbFrame

from vpc import face, pose
from vpc.helpers.geometry import rotation_matrix_between_vectors

__all__ = ['Estimator', 'Result', 'Result3d']


class VisualizationContext(TypedDict):
    gaze_draw_lines: bool
    gaze_line_color: Color
    gaze_line_thickness: int
    gaze_line_length: float


@dataclass(slots=True)
class Result:
    eyes: Float[torch.Tensor, 'n_detections 2 2']
    directions: Float[torch.Tensor, 'n_detections 2 2']

    device: torch.device
    """
    # Invariant:
    `self.eyes` and `self.directions` should be located on `self.device`
    """

    def draw(
        self,
        frame: ArrayRgbFrame,
        context: VisualizationContext,
    ) -> ArrayRgbFrame:
        if not context['gaze_draw_lines']:
            return frame

        color = context['gaze_line_color']
        thickness = context['gaze_line_thickness']
        line_length = context['gaze_line_length']

        starts = self.eyes.cpu().numpy()
        directions = self.directions.cpu().numpy()
        ends = starts + line_length * directions

        actor_starts: Int[numpy.ndarray, '2 2']
        actor_ends: Int[numpy.ndarray, '2 2']

        for actor_starts, actor_ends in zip(starts.astype(int), ends.astype(int)):
            opencv.line(
                frame,
                cast(opencv.typing.Point, actor_starts[0, :2]),
                cast(opencv.typing.Point, actor_ends[0, :2]),
                color,
                thickness,
            )
            opencv.line(
                frame,
                cast(opencv.typing.Point, actor_starts[1, :2]),
                cast(opencv.typing.Point, actor_ends[1, :2]),
                color,
                thickness,
            )
            opencv.circle(
                frame,
                cast(opencv.typing.Point, actor_starts[0, :2]),
                3,
                (0, 0, 255),
                3,
            )
            opencv.circle(
                frame,
                cast(opencv.typing.Point, actor_starts[1, :2]),
                3,
                (255, 0, 0),
                3,
            )

        return frame


@dataclass(slots=True)
class Result3d:
    eyes: Float[torch.Tensor, 'n_detections 2 3']
    directions: Float[torch.Tensor, 'n_detections 2 3']

    device: torch.device
    """
    # Invariant:
    `self.eyes` and `self.directions` should be located on `self.device`
    """

    # For numerical stability during transformation of a normalized `directions`
    __STABILIZING_MULTIPLIER: ClassVar[float] = 100.0

    def align(self, calibration: Calibration, poses: pose.Result3d) -> 'Result3d':
        projected = self.project(calibration)
        projected_starts = projected.eyes

        fx, fy = calibration.focal_length
        cx, cy = calibration.optical_center

        eye_indices = [
            pose.model.YoloKeypoint.LEFT_EYE,
            pose.model.YoloKeypoint.RIGHT_EYE,
        ]
        z = poses.keypoints[:, eye_indices, 2]

        new_eyes = torch.empty_like(self.eyes)
        new_eyes[..., 0] = z * (projected_starts[..., 0] - cx) / fx
        new_eyes[..., 1] = z * (projected_starts[..., 1] - cy) / fy
        new_eyes[..., 2] = z

        starts = self.eyes
        directions = self.directions

        new_directions = torch.zeros_like(self.directions)

        for i in range(len(starts)):
            eye_line = starts[i, 1, :] - starts[i, 0, :]
            left_eye_direction = directions[i, 0, :]
            right_eye_direction = directions[i, 1, :]

            left_gaze_rotation = rotation_matrix_between_vectors(
                eye_line,
                left_eye_direction,
            )
            if left_gaze_rotation is None:
                continue

            right_gaze_rotation = rotation_matrix_between_vectors(
                eye_line,
                right_eye_direction,
            )
            if right_gaze_rotation is None:
                continue

            new_eye_line = new_eyes[i, 1, :] - new_eyes[i, 0, :]
            new_directions[i, 0, :] = left_gaze_rotation @ new_eye_line
            new_directions[i, 1, :] = right_gaze_rotation @ new_eye_line

        new_directions /= new_directions.norm(2.0, dim=-1).unsqueeze(-1)

        return Result3d(new_eyes, new_directions, self.device)

    def project(self, calibration: Calibration) -> Result:
        cx, cy = calibration.optical_center
        fx, fy = calibration.focal_length

        # TODO(#60): remove the rescaling.
        eyes = self.eyes.clone() * 8.0 / 28.0

        z = eyes[..., -1]
        eyes[..., 0] *= fx
        eyes[..., 1] *= fy
        eyes[..., 0] /= z
        eyes[..., 1] /= z
        eyes[..., 0] += cx
        eyes[..., 1] += cy

        ends = self.eyes + self.__STABILIZING_MULTIPLIER * self.directions
        z = ends[..., -1]
        ends[..., 0] *= fx
        ends[..., 1] *= fy
        ends[..., 0] /= z
        ends[..., 1] /= z
        ends[..., 0] += cx
        ends[..., 1] += cy

        directions = ends - eyes
        directions_2d = directions[..., :-1]
        directions_2d /= directions_2d.norm(p=2, dim=2)

        return Result(eyes[..., :-1], directions_2d, self.device)


class Estimator:
    extractor: mini_face.gaze.Extractor
    average_direction: bool
    device: torch.device

    def __init__(
        self,
        calibration: Calibration,
        models_directory: Path,
        fps: int,
        analysis_mode: Literal['image', 'video'] = 'video',
        wild: bool = False,
        multiple_views: bool = False,
        limit_angles: bool = False,
        average_direction: bool = True,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device('cpu')
        self.average_direction = average_direction

        match analysis_mode:
            case 'image':
                mode = mini_face.PredictionMode.IMAGE
            case 'video':
                mode = mini_face.PredictionMode.VIDEO

        self.extractor = mini_face.gaze.Extractor(
            focal_length=calibration.focal_length,
            optical_center=calibration.optical_center,
            fps=50,
            models_directory=models_directory,
            mode=mode,
        )

    def predict(self, frame: TensorRgbFrame, faces: face.Result) -> Result3d | None:
        extractor = self.extractor

        eyes: list[Float[numpy.ndarray, '1 2 3'] | None] = []
        directions: list[Float[numpy.ndarray, '1 2 3'] | None] = []

        frame_array = numpy.ascontiguousarray(frame.cpu().permute((1, 2, 0)).numpy())

        box: UInt32[numpy.ndarray, '4']
        for box in faces.boxes.cpu().numpy().astype(numpy.uint32):
            detection = extractor.predict(frame_array, box)

            if detection is None:
                eyes.append(None)
                directions.append(None)
                continue

            actor_eyes = detection.eyes.mean(axis=0)
            actor_directions = detection.directions.mean(axis=0)

            if self.average_direction:
                average_direction = detection.directions[0].mean(axis=0)
                actor_directions[0, :] = average_direction
                actor_directions[1, :] = average_direction

            eyes.append(actor_eyes[numpy.newaxis, ...])
            directions.append(actor_directions[numpy.newaxis, ...])

        match (
            imputed_with_zeros_reference(eyes),
            imputed_with_zeros_reference(directions),
        ):
            case list(eyes_imputed), list(directions_imputed):
                return Result3d(
                    torch.from_numpy(
                        numpy.concatenate(
                            eyes_imputed,
                            axis=0,
                        )
                    ).to(self.device),
                    torch.from_numpy(
                        numpy.concatenate(
                            directions_imputed,
                            axis=0,
                        )
                    ).to(self.device),
                    self.device,
                )

            case _:
                return None

    def predict_batch(
        self,
        frames: UInt8[torch.Tensor, 'batch 3 height width'],
        faces: list[face.Result],
    ) -> list[Result3d | None]:
        return list(starmap(self.predict, zip(frames, faces)))


def imputed_with_zeros_reference[Shape: tuple[int, ...], Type: numpy.generic](
    batch: Sequence[numpy.ndarray[Shape, numpy.dtype[Type]] | None],
) -> list[numpy.ndarray[Shape, numpy.dtype[Type]]] | None:
    first_not_none = first_true(batch, None, lambda x: x is not None)

    if first_not_none is None:
        return None

    fill_element = numpy.zeros_like(first_not_none)

    return [element if element is not None else fill_element for element in batch]
