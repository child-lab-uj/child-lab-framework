from pathlib import Path

import cv2
import numpy as np
from tqdm import trange

from ..core.calibration import Calibration
from ..core.detection import chessboard
from ..core.video import Format, Input, Reader, Writer
from ..task.visualization import Configuration, Visualizer
from ..typing.array import FloatArray2, FloatArray3


# TODO: Implement procedures as classes with `Iterable` protocol
# to make them both usable with tqdm and exportable as purely programistic library elements
def run(
    video_source: Path,
    annotated_video_destination: Path,
    board_properties: chessboard.Properties,
    skip: int,
) -> Calibration:
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
        None,  # type: ignore
        properties=video_properties,
        configuration=Configuration(),
    )

    detector = chessboard.Detector(board_properties)

    inner_corners_per_row = board_properties.inner_corners_per_row
    inner_corners_per_column = board_properties.inner_corners_per_column
    square_size = board_properties.square_size

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0). ...,(6,5,0)
    chessboard_3d_model: FloatArray2 = np.zeros(
        (inner_corners_per_column * inner_corners_per_row, 3),
        np.float32,
    )

    chessboard_3d_model[:, :2] = (
        np.mgrid[0:inner_corners_per_row, 0:inner_corners_per_column].T.reshape(-1, 2)
        * square_size
    )

    object_points: list[FloatArray2] = []
    image_points: list[FloatArray3] = []

    for _ in trange(1, video_properties.length, skip, desc='Processing video'):
        frame = reader.read_skipping(skip)

        if frame is None:
            break

        result = detector.predict(frame)

        if result is None:
            continue

        writer.write(visualizer.annotate(frame, result))

        object_points.append(chessboard_3d_model)
        image_points.append(result.corners)

    # OpenCV needs at least 10 observations
    n_detections = len(image_points)
    if n_detections < 10:
        raise ValueError(
            f'At least 10 detections are required to perform calibration but only {n_detections} succeeded'
        )

    success, intrinsics, distortion, *_ = cv2.calibrateCamera(
        object_points,
        image_points,
        (video_properties.width, video_properties.height),
        None,  # type: ignore  # Opencv API...
        None,  # type: ignore
    )

    if not success:
        raise RuntimeError('OpenCV procedure ended unsuccessfully')

    # Explicitly cast to float to avoid storing NumPy scalars which serialize improperly
    return Calibration(
        optical_center=(float(intrinsics[0, 2]), float(intrinsics[1, 2])),
        focal_length=(float(intrinsics[0, 0]), float(intrinsics[1, 1])),
        distortion=np.squeeze(distortion),
    )
