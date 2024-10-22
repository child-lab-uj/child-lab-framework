from dataclasses import dataclass
from enum import IntEnum

import cv2 as opencv
import cv2.aruco as opencv_aruco
import numpy as np

from ....core.video import Calibration
from ....typing.array import FloatArray2, FloatArray3, IntArray1
from ....typing.video import Frame
from .. import transformation


# Can't stand that awful OpenCV's constants ported directly from C++; enum is the way
class Dictionary(IntEnum):
    ARUCO_ORIGINAL = opencv_aruco.DICT_ARUCO_ORIGINAL

    ARUCO_4X4_100 = opencv_aruco.DICT_4X4_100
    ARUCO_4X4_200 = opencv_aruco.DICT_4X4_250
    ARUCO_4X4_1000 = opencv_aruco.DICT_4X4_1000

    ARUCO_5X5_50 = opencv_aruco.DICT_5X5_50
    ARUCO_5X5_100 = opencv_aruco.DICT_5X5_100
    ARUCO_5X5_250 = opencv_aruco.DICT_5X5_250
    ARUCO_5X5_1000 = opencv_aruco.DICT_5X5_1000

    ARUCO_6X6_50 = opencv_aruco.DICT_6X6_50
    ARUCO_6X6_100 = opencv_aruco.DICT_6X6_100
    ARUCO_6X6_250 = opencv_aruco.DICT_6X6_250
    ARUCO_6X6_1000 = opencv_aruco.DICT_6X6_1000

    ARUCO_7X7_50 = opencv_aruco.DICT_7X7_50
    ARUCO_7X7_100 = opencv_aruco.DICT_7X7_100
    ARUCO_7X7_250 = opencv_aruco.DICT_7X7_250
    ARUCO_7X7_1000 = opencv_aruco.DICT_7X7_1000

    ARUCO_MIP_36H12 = opencv_aruco.DICT_ARUCO_MIP_36H12

    APRIL_16H5 = opencv_aruco.DICT_APRILTAG_16H5
    APRIL_25H9 = opencv_aruco.DICT_APRILTAG_25H9
    APRIL_36H10 = opencv_aruco.DICT_APRILTAG_36H10
    APRIL_36H11 = opencv_aruco.DICT_APRILTAG_36H11


@dataclass(frozen=True)
class Result:
    corners: FloatArray3
    ids: IntArray1
    transformations: list[transformation.Result | None]


class Detector:
    dictionary: Dictionary
    raw_dictionary: opencv_aruco.Dictionary
    detector: opencv_aruco.ArucoDetector

    marker_size: float
    marker_world_coordinates: FloatArray2

    calibration: Calibration

    def __init__(
        self,
        *,
        dictionary: Dictionary,
        detector_parameters: opencv_aruco.DetectorParameters,
        marker_size: float,
    ) -> None:
        self.dictionary = dictionary

        raw_dictionary = opencv_aruco.getPredefinedDictionary(dictionary.value)
        self.raw_dictionary = raw_dictionary

        self.detector = opencv_aruco.ArucoDetector(
            raw_dictionary,
            detector_parameters,
        )

        self.marker_size = marker_size

        self.marker_world_coordinates = np.array(
            [
                [-marker_size / 2.0, marker_size / 2.0, 0.0],
                [marker_size / 2.0, marker_size / 2.0, 0.0],
                [marker_size / 2.0, -marker_size / 2.0, 0.0],
                [-marker_size / 2.0, -marker_size / 2.0, 0.0],
            ],
            dtype=np.float32,
        )

    def predict(self, frame: Frame) -> Result | None:
        corners: list[FloatArray2]
        ids: IntArray1

        corners, ids, _rejected = self.detector.detectMarkers(frame)  # type: ignore # Please never write return types as protocols ;__;

        if len(corners) == 0 or len(ids) == 0:
            return None

        batched_corners = np.stack(corners)

        calibration = self.calibration
        intrinsics = calibration.intrinsics
        distortion = calibration.distortion

        marker_world_coordinates = self.marker_world_coordinates

        results = [
            # interpretation: rotate the camera so the marker will be in the center
            opencv.solvePnP(
                marker_world_coordinates,
                marker_corners,
                intrinsics,
                distortion,
                useExtrinsicGuess=True,
                flags=opencv.SOLVEPNP_IPPE_SQUARE,
            )
            for marker_corners in corners
        ]

        transformations = [
            transformation.Result(
                rotation,  # type: ignore
                translation,  # type: ignore
                intrinsics,
                distortion,
            )
            if success
            else None
            for success, rotation, translation in results
        ]

        return Result(batched_corners, ids, transformations)
