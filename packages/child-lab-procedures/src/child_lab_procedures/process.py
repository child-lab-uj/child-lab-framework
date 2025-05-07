from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import torch
from transformation_buffer.buffer import Buffer
from video_io.calibration import Calibration
from video_io.reader import Reader
from video_io.writer import Writer
from vpc import pose


@dataclass(slots=True)
class Configuration:
    device: torch.device
    batch_size: int
    max_detections: int

    yolo_checkpoint: Path
    mini_face_weights: Path


@dataclass(slots=True)
class VideoIoContext:
    name: str
    calibration: Calibration
    reader: Reader
    writer: Writer | None = None  # TODO: Visualization context


@dataclass(slots=True)
class VideoAnalysisContext:
    name: str
    calibration: Calibration
    reader: Reader

    pose_estimator: pose.Estimator

    writer: Writer | None = None  # TODO: Visualization context

    @classmethod
    def from_io(cls, io: VideoIoContext, configuration: Configuration) -> Self:
        pose_estimator = pose.Estimator(
            configuration.max_detections,
            configuration.yolo_checkpoint,
            configuration.device,
        )

        return cls(
            io.name,
            io.calibration,
            io.reader,
            pose_estimator,
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

                poses = context.pose_estimator.predict_batch(frame_batch)
                faces = []
                gazes = []

                if context.writer is not None:
                    items_to_draw = [
                        filter(None, frame_items)
                        for frame_items in zip(poses, faces, gazes)
                    ]
                    context.writer.write_batch(frame_batch, items_to_draw)

            if exhausted:
                break
