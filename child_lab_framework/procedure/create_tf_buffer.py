from dataclasses import dataclass
from typing import Optional
import cv2.aruco as aruco
import numpy as np
import torch
from ..core.transform_buffer import TransformBuffer
from ..core.video import Perspective, Reader, Calibration
from ..core.algebra import kabsch
from ..task.camera.detection.marker import (
    Detector as MarkerDetector,
    Dictionary as ArucoDictionary,
)
from ..task.depth.depth import Estimator as DepthEstimator
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

MARKER_PREFIX = '_MARKER'


@dataclass
class Input:
    source: str
    frame_of_reference: str
    calibration: Calibration


@dataclass
class Config:
    aruco_marker_size: float
    aruco_dictionary: ArucoDictionary
    aruco_detector_params: aruco.DetectorParameters = aruco.DetectorParameters()


def create_tf_buffer(
    inputs: list[Input],
    config: Config,
    device: torch.device,
    initial_tf_state: Optional[TransformBuffer.State] = None,
) -> TransformBuffer.State:
    assert len(inputs) > 1

    def graph_ready(tf_buffer: TransformBuffer):
        frames_of_reference = [input.frame_of_reference for input in inputs]
        frames_of_reference = [
            frame for frame in frames_of_reference if not frame.startswith(MARKER_PREFIX)
        ]
        first_frame = frames_of_reference.pop()
        return all(
            [
                tf_buffer.transform_available(first_frame, frame)
                for frame in frames_of_reference
            ]
        )

    # TODO: Dumping AruDice to config
    if initial_tf_state:
        tf_buffer = TransformBuffer.from_state(initial_tf_state)
    else:
        tf_buffer = TransformBuffer()

    readers = [
        Reader(input.source, perspective=Perspective.OTHER, batch_size=1)
        for input in inputs
    ]

    detector = MarkerDetector(
        dictionary=config.aruco_dictionary,
        detector_parameters=config.aruco_detector_params,
        marker_size=config.aruco_marker_size,
    )

    executor = ThreadPoolExecutor(max_workers=8)
    estimator = DepthEstimator(
        executor=executor,
        device=device,
        input=readers[0].properties,
    )

    while True:
        frames = [(input, reader.read()) for input, reader in zip(inputs, readers)]
        frames = list(filter(lambda f: f[1] is not None, frames))

        if not any(frames):
            break

        for input, rgb_frame in tqdm(frames, 'Processing frames'):
            detector.calibration = input.calibration
            result = detector.predict(rgb_frame)

            if not result:
                continue

            depth = estimator.predict(rgb_frame)
            points = input.calibration.depth_to_3D(depth)

            marker_local_coordinates = np.array(
                [
                    [
                        -config.aruco_marker_size / 2.0,
                        config.aruco_marker_size / 2.0,
                        0.0,
                    ],
                    [config.aruco_marker_size / 2.0, config.aruco_marker_size / 2.0, 0.0],
                    [
                        config.aruco_marker_size / 2.0,
                        -config.aruco_marker_size / 2.0,
                        0.0,
                    ],
                    [
                        -config.aruco_marker_size / 2.0,
                        -config.aruco_marker_size / 2.0,
                        0.0,
                    ],
                ],
                dtype=np.float32,
            )

            for corners, id in zip(result.corners, result.ids):
                corners_3D = np.array(
                    [
                        points[int(corner_y), int(corner_x)]
                        for (corner_x, corner_y) in corners.squeeze()
                    ],
                    dtype=np.float32,
                )
                R, t = kabsch(corners_3D.T, marker_local_coordinates.T)
                tf_buffer.add_transform(
                    input_frame=input.frame_of_reference,
                    output_frame=f'{MARKER_PREFIX}_{id}',
                    rotation=R,
                    translation=t,
                )

        # End early if the graph is ready (i.e. transforms between all cameras)
        if graph_ready(tf_buffer):
            break

    return tf_buffer.get_state()
