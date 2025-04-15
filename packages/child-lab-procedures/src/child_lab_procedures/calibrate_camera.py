# +-------------------------------------------------------------------------------------+
# | Calibration algorithm inspired by:                                                  |
# | Repository: https://github.com/ros-perception/image_pipeline                        |
# | File: `image_pipeline/camera_calibration/src/camera_calibration/mono_calibrator.py` |                                                                             |
# | Commit: 722ca08b98f37b7b148d429753da133ff1e2c7cf                                    |
# +-------------------------------------------------------------------------------------+

import random
from collections.abc import Callable
from dataclasses import dataclass
from textwrap import dedent
from typing import Self

import cv2 as opencv
import numpy
from jaxtyping import Float
from marker_detection import chessboard
from marker_detection.chessboard import (
    AggregatedDetectionDetails,
    BoardProperties,
    DetectionDetails,
    Detector,
    VisualizationContext,
)
from video_io.calibration import Calibration
from video_io.reader import Reader
from video_io.writer import Writer

from child_lab_procedures.garbage_collection import no_garbage_collection


@dataclass(slots=True)
class Configuration:
    board_properties: BoardProperties
    max_samples: int | None = None
    max_speed: float = float('inf')
    min_distance: float = 0.3


@dataclass(slots=True)
class VideoIoContext:
    name: str
    reader: Reader
    writer: Writer[VisualizationContext] | None = None


@dataclass(slots=True)
class VideoAnalysisContext:
    name: str
    reader: Reader
    detector: Detector
    writer: Writer[VisualizationContext] | None = None

    @classmethod
    def from_io(cls, io: VideoIoContext, configuration: Configuration) -> Self:
        detector = Detector(configuration.board_properties)

        return cls(
            io.name,
            io.reader,
            detector,
            io.writer,
        )


@dataclass(slots=True, repr=False)
class Result:
    calibration: Calibration
    reprojection_error: float

    samples: int
    speed_rejections: int
    similarity_rejections: int

    average_metrics: DetectionDetails
    progress_metrics: DetectionDetails

    def __repr__(self) -> str:
        calibration = self.calibration
        average_metrics = self.average_metrics
        progress_metrics = self.progress_metrics

        def format_float_tuple(input: tuple[float, ...], format_specifier: str) -> str:
            elements = ', '.join(x.__format__(format_specifier) for x in input)
            return f'({elements})'

        def percent(value: float) -> str:
            return f'{value * 100.0:.2f}%'

        return dedent(
            f"""\
            Result:
              samples: {self.samples}
              speed_rejections: {self.speed_rejections}
              similarity_rejections: {self.similarity_rejections}
              reprojection_error: {self.reprojection_error:.3e}

              calibration: Calibration:
                focal_length: {format_float_tuple(calibration.focal_length, '.2f')}
                optical_center: {format_float_tuple(calibration.optical_center, '.2f')}
                distortion: {format_float_tuple(calibration.distortion, '.3e')}

              average_metrics: DetectionProperties:
                area: {average_metrics.area:.3e}
                skew: {average_metrics.skew:.3e}
                x_offset: {average_metrics.x_offset:.3e}
                y_offset: {average_metrics.y_offset:.3e}
                perspective_offset: {average_metrics.perspective_offset:.3e}

              progress_metrics: DetectionProperties:
                skew: {percent(progress_metrics.skew)}
                x_offset: {percent(progress_metrics.x_offset)}
                y_offset: {percent(progress_metrics.y_offset)}
                perspective_offset: {percent(progress_metrics.perspective_offset)}
            """
        )


@dataclass(slots=True)
class SamplingSummary:
    """
    Container which stores intermediate results required to compute the calibration.
    """

    image_size: tuple[int, int]
    board_properties: BoardProperties

    samples: list[Float[numpy.ndarray, 'n_points 1 2']]
    metrics: list[DetectionDetails]
    speed_rejections: int
    similarity_rejections: int

    def calibrate(self) -> Result:
        """
        Compute the calibration based on the collected samples.
        """

        board_3d_points = self.board_properties.rigid_model
        samples = self.samples

        reprojection_error, intrinsics, distortion, *_ = opencv.calibrateCamera(
            [board_3d_points for _ in range(len(samples))],
            samples,
            (self.image_size[1], self.image_size[0]),
            numpy.eye(3, 3, dtype=numpy.float32),
            numpy.zeros(5, dtype=numpy.float32),
        )
        assert distortion.shape == (5, 1)
        assert intrinsics.shape == (3, 3)

        calibration = Calibration(
            focal_length=(float(intrinsics[0, 0]), float(intrinsics[1, 1])),
            optical_center=(float(intrinsics[0, 2]), float(intrinsics[1, 2])),
            distortion=tuple(distortion.flatten().tolist()),
        )

        aggregated_metrics = AggregatedDetectionDetails(self.metrics)
        average_metrics = aggregated_metrics.mean()
        progress_metrics = aggregated_metrics.progress()

        return Result(
            calibration,
            reprojection_error,
            len(samples),
            self.speed_rejections,
            self.similarity_rejections,
            average_metrics,
            progress_metrics,
        )


class Procedure:
    configuration: Configuration
    context: VideoAnalysisContext

    def __init__(
        self,
        configuration: Configuration,
        context: VideoIoContext,
    ) -> None:
        self.configuration = configuration
        self.context = VideoAnalysisContext.from_io(context, configuration)

    def length_estimate(self) -> int:
        return self.context.reader.metadata.frames

    @no_garbage_collection()
    def run(self, on_step: Callable[[], object] = lambda: None) -> SamplingSummary | None:
        reader = self.context.reader
        detector = self.context.detector
        writer = self.context.writer

        frames_since_previous_result = 0
        time_delta = 1.0 / reader.metadata.fps
        max_speed = self.configuration.max_speed
        min_distance = self.configuration.min_distance
        speed_rejections = 0
        similarity_rejections = 0

        image_points: list[Float[numpy.ndarray, 'n_corners 1 2']] = []
        previous_result: chessboard.Result | None = None
        metrics: list[chessboard.DetectionDetails] = []

        while (tensor_frame := reader.read()) is not None:
            on_step()

            result = detector.predict(tensor_frame.permute((1, 2, 0)).numpy())
            if result is None:
                frames_since_previous_result += 1
                continue

            speed = (
                result.average_speed(
                    previous_result,
                    time_delta * frames_since_previous_result,
                )
                if previous_result is not None
                else 0.0
            )
            frames_since_previous_result = 1
            previous_result = result

            if speed >= max_speed:
                speed_rejections += 1
                continue

            result_metrics = result.details

            closest_sample_distance = min(
                (result_metrics.distance(other) for other in metrics),
                default=float('inf'),
            )
            if closest_sample_distance <= min_distance:
                similarity_rejections += 1
                continue

            metrics.append(result_metrics)
            image_points.append(result.corners)

            if writer is not None:
                writer.write(tensor_frame, [result])

        n_samples = len(image_points)
        max_samples = self.configuration.max_samples

        if max_samples is not None and n_samples > max_samples:
            n_samples = max_samples
            indices = list(range(n_samples))
            random_samples = random.sample(indices, max_samples)
            image_points = [image_points[i] for i in random_samples]
            metrics = [metrics[i] for i in random_samples]

        return SamplingSummary(
            (reader.metadata.height, reader.metadata.width),
            self.configuration.board_properties,
            image_points,
            metrics,
            speed_rejections,
            similarity_rejections,
        )
