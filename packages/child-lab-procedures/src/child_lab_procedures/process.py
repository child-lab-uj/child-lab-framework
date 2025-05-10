from collections.abc import Callable
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Any, Self

import torch
from icecream import ic
from transformation_buffer.buffer import Buffer
from video_io.calibration import Calibration
from video_io.reader import Reader
from video_io.writer import Writer
from vpc import face, gaze, pose

from .imputation import imputed_with_closest_known_reference as imputed

__all__ = ['Configuration', 'VisualizationContext', 'VideoIoContext', 'Procedure']


@dataclass(slots=True)
class Configuration:
    device: torch.device
    batch_size: int
    max_detections: int

    yolo_checkpoint: Path
    mini_face_models_directory: Path

    face_confidence_threshold: float
    face_suppression_threshold: float


class VisualizationContext(
    pose.VisualizationContext,
    face.VisualizationContext,
    gaze.VisualizationContext,
): ...


@dataclass(slots=True)
class VideoIoContext:
    name: str
    calibration: Calibration
    reader: Reader
    writer: Writer[VisualizationContext] | None = None


@dataclass(slots=True)
class VideoAnalysisContext:
    name: str
    calibration: Calibration
    reader: Reader

    pose_estimator: pose.Estimator
    face_estimator: face.Estimator
    gaze_estimator: gaze.Estimator

    writer: Writer[VisualizationContext] | None = None

    @classmethod
    def from_io(cls, io: VideoIoContext, configuration: Configuration) -> Self:
        pose_estimator = pose.Estimator(
            configuration.max_detections,
            configuration.yolo_checkpoint,
            configuration.device,
        )
        face_estimator = face.Estimator(
            configuration.face_confidence_threshold,
            configuration.face_suppression_threshold,
            configuration.device,
        )
        gaze_estimator = gaze.Estimator(
            io.calibration,
            configuration.mini_face_models_directory,
        )

        return cls(
            io.name,
            io.calibration,
            io.reader,
            pose_estimator,
            face_estimator,
            gaze_estimator,
            io.writer,
        )


class Procedure:
    """
    Computes **all** possible features eagerly.
    """

    contexts: list[VideoAnalysisContext]
    transformation_buffer: Buffer[str]

    configuration: Configuration

    def __init__(
        self,
        configuration: Configuration,
        io_contexts: list[VideoIoContext],
        transformation_buffer: Buffer[str],
    ) -> None:
        self.configuration = configuration
        self.contexts = [
            VideoAnalysisContext.from_io(io, configuration) for io in io_contexts
        ]
        self.transformation_buffer = transformation_buffer

    def run(self, on_step: Callable[[], Any]) -> None:
        while True:
            if on_step is not None:
                on_step()

            exhausted = True

            for context in self.contexts:
                frame_batch = context.reader.read_batch(self.configuration.batch_size)
                if frame_batch is None:
                    continue

                exhausted = False

                ic(context.name)
                ic(context.calibration)

                poses = imputed(context.pose_estimator.predict_batch(frame_batch))
                faces = (
                    imputed(context.face_estimator.predict_batch(frame_batch, poses))
                    if poses is not None
                    else None
                )
                gazes = (
                    imputed(context.gaze_estimator.predict_batch(frame_batch, faces))
                    if faces is not None
                    else None
                )
                projected_gazes = (
                    [gaze.project(context.calibration) for gaze in gazes]
                    if gazes is not None
                    else None
                )
                ic(projected_gazes)

                if context.writer is not None:
                    items_to_draw = [
                        filter(None, frame_items)
                        for frame_items in zip(
                            poses or repeat(None),
                            faces or repeat(None),
                            projected_gazes or repeat(None),
                        )
                    ]
                    context.writer.write_batch(frame_batch, items_to_draw)

            if exhausted:
                break
