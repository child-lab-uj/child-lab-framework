import gc
import random
import typing
from collections.abc import Iterable
from contextlib import ContextDecorator
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from types import TracebackType
from typing import Literal, Self

import cv2
import numpy as np
from tqdm import trange

from ..core.calibration import Calibration
from ..core.video import Format, Input, Reader, Writer
from ..task.camera.detection import chessboard
from ..task.visualization import Configuration, Visualizer
from ..typing.array import FloatArray1

# +-------------------------------------------------------------------------------------+
# | Calibration algorithm inspired by:                                                  |
# | Repository: https://github.com/ros-perception/image_pipeline                        |
# | File: `image_pipeline/camera_calibration/src/camera_calibration/mono_calibrator.py` |                                                                             |
# | Commit: 722ca08b98f37b7b148d429753da133ff1e2c7cf                                    |
# +-------------------------------------------------------------------------------------+


@dataclass(frozen=True, slots=True)
class EliminationProperties:
    max_samples: int | None = None
    max_speed: float = float('inf')
    min_distance: float = 0.3


@dataclass(frozen=True, slots=True, repr=False)
class Result:
    samples: int
    speed_rejections: int
    similarity_rejections: int
    reprojection_error: float
    calibration: Calibration
    average_metrics: chessboard.DetectionProperties
    progress_metrics: chessboard.DetectionProperties

    def __repr__(self) -> str:
        calibration = self.calibration
        average_metrics = self.average_metrics
        progress_metrics = self.progress_metrics

        def float_tuple(input: Iterable[float], format_specifier: str) -> str:
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
                focal_length: {float_tuple(calibration.focal_length, '.2f')}
                optical_center: {float_tuple(calibration.optical_center, '.2f')}
                distortion: {float_tuple(calibration.distortion.tolist(), '.3e')}

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


class no_garbage_collection(ContextDecorator):
    def __init__(self) -> None:
        super().__init__()

    def __enter__(self) -> Self:
        gc.disable()
        return self

    def __exit__(
        self,
        _exception_kind: type | None,
        exception: Exception | None,
        _traceback: TracebackType | None,
        **_: object,
    ) -> Literal[False]:
        gc.enable()

        if exception is not None:
            raise exception

        return False


# TODO: Implement procedures as classes with `Iterable` protocol
# to make them both usable with tqdm and exportable as purely programistic library elements
@no_garbage_collection()
def run(
    video_source: Path,
    annotated_video_destination: Path,
    board_properties: chessboard.BoardProperties,
    elimination_properties: EliminationProperties,
) -> Result:
    reader = Reader(
        Input(video_source.name, video_source, None),
        batch_size=1,
    )

    video_properties = reader.properties

    writer = Writer(
        annotated_video_destination,
        video_properties,
        output_format=Format.MP4,
    )

    visualizer = Visualizer(
        properties=video_properties,
        configuration=Configuration(),
    )

    detector = chessboard.Detector(board_properties)

    metrics: list[chessboard.DetectionProperties] = []

    image_points: list[
        np.ndarray[tuple[int, Literal[1], Literal[2]], np.dtype[np.float32]]
    ] = []

    previous_result: chessboard.Result | None = None
    frames_since_previous_result = 0

    time_delta = 1.0 / reader.properties.fps
    max_speed = elimination_properties.max_speed
    min_distance = elimination_properties.min_distance

    progress_bar = trange(1, video_properties.length, desc='Processed frames')
    progress_bar.reset()
    progress_bar.refresh()

    result_count_bar = trange(
        1,
        video_properties.length,
        desc='Important samples',
    )
    result_count_bar.reset()
    result_count_bar.refresh()

    speed_rejections = 0
    similarity_rejections = 0

    while (frame := reader.read()) is not None:
        progress_bar.update()

        result = detector.predict(frame)

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

        result_metrics = result.detection_properties
        closest_sample_distance = min(
            (result_metrics.distance(other) for other in metrics),
            default=float('inf'),
        )

        if closest_sample_distance <= min_distance:
            similarity_rejections += 1
            continue

        result_count_bar.update()

        metrics.append(result_metrics)
        image_points.append(result.corners)

        writer.write(visualizer.annotate(frame, result))

    progress_bar.refresh()
    result_count_bar.refresh()

    progress_bar.close()
    result_count_bar.close()

    # OpenCV needs at least 10 observations
    n_samples = len(image_points)
    if n_samples < 10:
        raise ValueError(
            f'At least 10 detections are required to perform calibration but only {n_samples} have been collected'
        )

    max_samples = elimination_properties.max_samples
    if max_samples is not None and n_samples > max_samples:
        indices = list(range(n_samples))

        random_samples = random.sample(indices, max_samples)

        image_points = [image_points[i] for i in random_samples]
        metrics = [metrics[i] for i in random_samples]

        n_samples = max_samples

    print(f'Calibrating with {n_samples} samples...')

    board_3d_points = board_properties.rigid_model

    reprojection_error, intrinsics, distortion_dirty, *_ = cv2.calibrateCamera(
        [board_3d_points for _ in image_points],
        image_points,
        (video_properties.width, video_properties.height),
        np.eye(3, 3, dtype=np.float32),
        np.zeros(5, dtype=np.float32),
    )

    distortion = typing.cast(FloatArray1, np.squeeze(distortion_dirty))

    # Explicitly cast to float to avoid storing NumPy scalars which serialize improperly
    calibration = Calibration(
        optical_center=(float(intrinsics[0, 2]), float(intrinsics[1, 2])),
        focal_length=(float(intrinsics[0, 0]), float(intrinsics[1, 1])),
        distortion=distortion,
    )

    aggregated_metrics = chessboard.AggregatedDetectionProperties(metrics)
    average_metrics = aggregated_metrics.mean()
    progress_metrics = aggregated_metrics.progress()

    return Result(
        n_samples,
        speed_rejections,
        similarity_rejections,
        reprojection_error,
        calibration,
        average_metrics,
        progress_metrics,
    )
