from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Self

import cv2
import torch
from marker_detection.aruco import (
    Detector,
    Dictionary,
    RigidModel,
    VisualizationContext,
)
from transformation_buffer import Buffer
from transformation_buffer.rigid_model import Cube
from transformation_buffer.transformation import Transformation
from video_io import Calibration, Reader, Writer
from video_io.frame import ArrayRgbFrame

from child_lab_procedures.garbage_collection import no_garbage_collection


@dataclass(frozen=True)
class Configuration:
    model: RigidModel
    dictionary: Dictionary
    detector_parameters: cv2.aruco.DetectorParameters = cv2.aruco.DetectorParameters()
    arudice: list[Cube[str]] = field(default_factory=list)


@dataclass
class VideoIoContext:
    name: str
    calibration: Calibration
    reader: Reader
    writer: Writer[VisualizationContext] | None = None


@dataclass
class VideoAnalysisContext:
    name: str
    calibration: Calibration
    reader: Reader
    detector: Detector
    writer: Writer[VisualizationContext] | None = None

    @classmethod
    def from_io(cls, io: VideoIoContext, configuration: Configuration) -> Self:
        detector = Detector(
            configuration.model,
            configuration.dictionary,
            configuration.detector_parameters,
            io.calibration.intrinsics_matrix().numpy(),
            io.calibration.distortion_vector().numpy(),
        )

        return cls(
            io.name,
            io.calibration,
            io.reader,
            detector,
            io.writer,
        )


class Procedure:
    MARKER_PREFIX = 'marker'

    contexts: list[VideoAnalysisContext]
    buffer: Buffer[str]

    def __init__(
        self,
        configuration: Configuration,
        contexts: list[VideoIoContext],
    ) -> None:
        self.contexts = [
            VideoAnalysisContext.from_io(ctx, configuration) for ctx in contexts
        ]

        buffer = Buffer((context.name for context in contexts))

        for arudie in configuration.arudice:
            buffer.add_object(arudie)

        self.buffer = buffer

    def length_estimate(self) -> int:
        return max((ctx.reader.metadata.frames for ctx in self.contexts), default=0)

    @no_garbage_collection()
    def run(self, on_step: Callable[[], object] = lambda: None) -> Buffer[str] | None:
        buffer = self.buffer

        while True:
            if on_step is not None:
                on_step()

            exhausted = True

            for context in self.contexts:
                frame_tensor = context.reader.read()

                if frame_tensor is None:
                    continue

                exhausted = False

                frame: ArrayRgbFrame = frame_tensor.permute((1, 2, 0)).numpy()
                markers = context.detector.predict(frame)

                if markers is None:
                    continue

                if context.writer is not None:
                    context.writer.write(frame_tensor, [markers])

                id: int
                for id, marker_transformation in zip(
                    markers.ids,
                    markers.transformations,
                ):
                    buffer[f'{Procedure.MARKER_PREFIX}_{int(id)}', context.name] = (
                        Transformation.from_parts(
                            torch.from_numpy(marker_transformation.rotation),
                            torch.from_numpy(marker_transformation.translation),
                        ).to(dtype=torch.float64)
                    )

            if buffer.connected:
                return buffer

            if exhausted:
                break

        return None
