from dataclasses import dataclass
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from ..core import transformation
from ..core.calibration import Calibration
from ..core.detection import marker
from ..core.video import Format, Input, Reader, Writer
from ..task import visualization

MARKER_PREFIX = 'marker'


@dataclass(frozen=True)
class Config:
    model: marker.RigidModel
    dictionary: marker.Dictionary
    detector_parameters: cv2.aruco.DetectorParameters = cv2.aruco.DetectorParameters()


# TODO: Implement procedures as classes with `Iterable` protocol
# to make them both usable with tqdm and exportable as purely programistic library elements
def run(
    video_sources: list[Path],
    video_destinations: list[Path],
    calibrations: list[Calibration],
    config: Config,
    device: torch.device,
) -> transformation.Buffer[str]:
    if len(video_sources) < 2:
        raise ValueError('At least two inputs are required to estimate transformations')

    # TODO: Dump AruDice to config
    buffer = transformation.Buffer({input.stem for input in video_sources})

    readers = [
        Reader(Input(input.stem, input, calibration), batch_size=1)
        for input, calibration in zip(video_sources, calibrations)
    ]

    writers = [
        Writer(destination, reader.properties, output_format=Format.MP4)
        for destination, reader in zip(video_destinations, readers)
    ]

    detector = marker.Detector(
        model=config.model,
        dictionary=config.dictionary,
        detector_parameters=config.detector_parameters,
    )

    visualizer = visualization.Visualizer(
        None,  # type: ignore
        properties=readers[0].properties,
        configuration=visualization.Configuration(),
    )

    while not buffer.connected:
        views = [
            (reader, writer, frame)
            for reader, writer in zip(readers, writers)
            if (frame := reader.read()) is not None
        ]

        if len(views) == 0:
            break

        for reader, writer, frame in tqdm(views, 'Processing frames'):
            detector.calibration = reader.properties.calibration
            markers = detector.predict(frame)

            if markers is None:
                continue

            writer.write(visualizer.annotate(frame, markers))

            id: int
            for id, marker_transformation in zip(markers.ids, markers.transformations):
                if marker_transformation is None:
                    continue

                buffer[f'{MARKER_PREFIX}_{int(id)}', reader.input.name] = (
                    marker_transformation
                )

    return buffer
