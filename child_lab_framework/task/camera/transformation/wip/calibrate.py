from dataclasses import dataclass
from typing import List
import cv2
import numpy as np


@dataclass
class Camera:
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    rotation_vectors: List[np.ndarray]
    translation_vectors: List[np.ndarray]


def nth_frame(video_path: str, n: int):
    """
    Generator that yields every N'th frame of a video.

    :param video_path: Path to the video file.
    :param n: The interval of frames to skip.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f'Error opening video file: {video_path}')

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % n == 0:
            yield frame

        frame_index += 1

    cap.release()


def calibrate_camera(video_path: str, frame_skip: int) -> Camera:
    """
    Calibrate the camera using frames from a video.

    :param video_path: Path to the video file.
    :param frame_skip: The interval of frames to skip.
    """
    # Termination criteria for corner sub-pixel refinement
    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0). ...,(6,5,0)
    object_points_template = np.zeros((6 * 7, 3), np.float32)
    object_points_template[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    object_points = []  # 3D points in real world space
    image_points = []  # 2D points in image plane

    i = 0
    for frame in nth_frame(video_path, frame_skip):
        print(i)
        i += 1

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        found, corners = cv2.findChessboardCorners(gray_frame, (7, 6), None)

        # If found, add object points and image points (after refining them)
        if found:
            object_points.append(object_points_template)

            refined_corners = cv2.cornerSubPix(
                gray_frame, corners, (11, 11), (-1, -1), termination_criteria
            )
            image_points.append(refined_corners)

    print(object_points, image_points)

    # Calibrate the camera
    ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors = (
        cv2.calibrateCamera(
            object_points, image_points, gray_frame.shape[::-1], None, None
        )
    )

    if not ret:
        raise RuntimeError('Camera calibration failed')

    return Camera(
        camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors
    )


if __name__ == '__main__':
    video_path = 'example.avi'
    frame_skip = 6  # Change this to the desired interval

    camera = calibrate_camera(video_path, frame_skip)
    print('Camera matrix:\n', camera.camera_matrix)
    print('Distortion coefficients:\n', camera.distortion_coefficients)
    print('Rotation vectors:\n', camera.rotation_vectors)
    print('Translation vectors:\n', camera.translation_vectors)
