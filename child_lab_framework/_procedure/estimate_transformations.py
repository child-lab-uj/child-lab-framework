from dataclasses import dataclass

import cv2
import torch
from tqdm import tqdm

from ..core import transformation
from ..core.detection import marker
from ..core.video import Input, Reader

MARKER_PREFIX = 'marker'


@dataclass(frozen=True)
class Config:
    model: marker.RigidModel
    dictionary: marker.Dictionary
    detector_parameters: cv2.aruco.DetectorParameters = cv2.aruco.DetectorParameters()


# TODO: Implement procedures as classes with `Iterable` protocol
# to make them both usable with tqdm and exportable as purely programistic library elements
def run(
    inputs: list[Input], config: Config, device: torch.device
) -> transformation.Buffer[str]:
    if len(inputs) < 2:
        raise ValueError('At least two inputs are required to estimate transformations')

    # TODO: Dump AruDice to config
    buffer = transformation.Buffer({input.name for input in inputs})

    readers = [Reader(input, batch_size=1) for input in inputs]

    marker_model = config.model

    marker_detector = marker.Detector(
        model=marker_model,
        dictionary=config.dictionary,
        detector_parameters=config.detector_parameters,
    )

    while not buffer.connected:
        frames = [
            (reader, frame) for reader in readers if (frame := reader.read()) is not None
        ]

        if len(frames) == 0:
            break

        for reader, frame in tqdm(frames, 'Processing frames'):
            marker_detector.calibration = reader.properties.calibration
            markers = marker_detector.predict(frame)

            if markers is None:
                continue

            id: int
            for id, marker_transformation in zip(markers.ids, markers.transformations):
                if marker_transformation is None:
                    continue

                buffer[f'{MARKER_PREFIX}_{int(id)}', reader.input.name] = (
                    marker_transformation
                )

    return buffer
