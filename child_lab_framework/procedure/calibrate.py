import cv2
import numpy as np
from tqdm import tqdm

from ..core.video import Calibration, Perspective, Reader
from ..task.camera.detection import chessboard


def calibrate(
    source: str,
    inner_corners_per_row: int,
    inner_corners_per_column: int,
    frame_skip: int = 60,
    square_size: int = 1,
):
    reader = Reader(
        source,
        perspective=Perspective.OTHER,
        batch_size=1,
    )

    detector = chessboard.Detector(
        inner_corners_per_row=inner_corners_per_row,
        inner_corners_per_column=inner_corners_per_column,
    )

    def skip_frame_iterator():
        i = 0
        while (frame := reader.read()) is not None:
            if i % frame_skip == 0:
                yield frame
            i += 1

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0). ...,(6,5,0)
    object_points_template = np.zeros(
        (inner_corners_per_column * inner_corners_per_row, 3), np.float32
    )
    object_points_template[:, :2] = (
        np.mgrid[0:inner_corners_per_row, 0:inner_corners_per_column].T.reshape(-1, 2)
        * square_size
    )

    image_points = []
    object_points = []

    for frame in tqdm(skip_frame_iterator(), desc=f'Processing frames ({frame_skip}/it)'):
        result = detector.predict(frame)
        if result:
            object_points.append(object_points_template)
            image_points.append(result.corners)

    # Trust the process :praying_hands:
    calibration_shape = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).shape[::-1]

    _ret, intrinsics, distortion, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points, image_points, calibration_shape, None, None
    )

    return Calibration(
        optical_center=(intrinsics[0, 2], intrinsics[1, 2]),
        focal_length=(intrinsics[0, 0], intrinsics[1, 1]),
        distortion=distortion,
    )
