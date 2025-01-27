from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from tqdm import trange

from ..core import transformation
from ..core.calibration import Calibration
from ..core.video import Format, Input, Reader, Writer
from ..task import visualization
from ..task.camera.detection import marker

MARKER_PREFIX = 'marker'


@dataclass(frozen=True)
class Configuration:
    model: marker.RigidModel
    dictionary: marker.Dictionary
    detector_parameters: cv2.aruco.DetectorParameters = cv2.aruco.DetectorParameters()


# TODO: Implement procedures as classes with `Iterable` protocol
# to make them both usable with tqdm and exportable as purely programistic library elements
def run(
    video_sources: list[Path],
    video_destinations: list[Path],
    calibrations: list[Calibration],
    skip: int,
    configuration: Configuration,
    device: torch.device,
) -> transformation.Buffer[str]:
    if len(video_sources) < 2:
        raise ValueError('At least two inputs are required to estimate transformations')

    # TODO: Dump AruDice to config
    buffer = transformation.Buffer({input.stem for input in video_sources})

    readers = [
        Reader(
            Input(input.stem, input, calibration),
            height=1080,
            width=1920,
            batch_size=1,
        )
        for input, calibration in zip(video_sources, calibrations)
    ]

    for reader in readers:
        reader.read_skipping(skip)

    writers = [
        Writer(destination, reader.properties, output_format=Format.MP4)
        for destination, reader in zip(video_destinations, readers)
    ]

    detector = marker.Detector(
        model=configuration.model,
        dictionary=configuration.dictionary,
        detector_parameters=configuration.detector_parameters,
    )

    visualizer = visualization.Visualizer(
        properties=readers[0].properties,
        configuration=visualization.Configuration(),
    )

    camera_progress_bar = trange(len(readers), desc='Processing cameras')
    frame_progress_bar = trange(
        min(
            (reader.properties.length - skip for reader in readers),
            default=0,
        ),
        desc='Processing frames',
    )

    while not buffer.connected:
        frame_progress_bar.update()

        views = [
            (reader, writer, frame)
            for reader, writer in zip(readers, writers)
            if (frame := reader.read()) is not None
        ]

        if len(views) == 0:
            break

        camera_progress_bar.refresh()
        camera_progress_bar.reset()

        for reader, writer, frame in views:
            camera_progress_bar.update()

            detector.calibration = reader.properties.calibration
            markers = detector.predict(frame)

            if markers is None:
                continue

            writer.write(visualizer.annotate(frame, markers))

            id: int
            for id, marker_transformation in zip(markers.ids, markers.transformations):
                buffer[f'{MARKER_PREFIX}_{int(id)}', reader.input.name] = (
                    marker_transformation
                )

    return buffer
