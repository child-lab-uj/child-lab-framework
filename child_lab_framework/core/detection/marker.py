from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Self

import cv2 as opencv
import cv2.aruco as opencv_aruco
import numpy as np

from ...typing.array import FloatArray2, FloatArray3, IntArray1
from ...typing.video import Frame
from .. import serialization, transformation
from ..calibration import Calibration
from ..video import Properties


# Can't stand that awful OpenCV's constants ported directly from C++; enum is better
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

    @classmethod
    def parse(cls, input: str) -> 'Dictionary | None':
        return _DICTIONARY_NAMES_TO_VARIANTS.get(input)

    def serialize(self) -> dict[str, serialization.Value]:
        return {'variant': _DICTIONARY_VARIANTS_TO_NAMES[self]}

    @classmethod
    def deserialize(cls, data: dict[str, serialization.Value]) -> Self:
        match data:
            case {'variant': str(variant), **_other}:
                instance = _DICTIONARY_NAMES_TO_VARIANTS.get(variant)

                if instance is None:
                    raise serialization.DeserializeError(
                        f'Unknown Dictionary name: {variant}'
                    )

                return instance  # type: ignore

            case other:
                raise serialization.DeserializeError(
                    f'Expected dictionary with variant: str, got {other}'
                )


_DICTIONARY_NAMES_TO_VARIANTS: dict[str, Dictionary] = {
    'aruco-original': Dictionary.ARUCO_ORIGINAL,
    'aruco-4x4-100': Dictionary.ARUCO_4X4_100,
    'aruco-4x4-200': Dictionary.ARUCO_4X4_200,
    'aruco-4x4-1000': Dictionary.ARUCO_4X4_1000,
    'aruco-5x5-50': Dictionary.ARUCO_5X5_50,
    'aruco-5x5-100': Dictionary.ARUCO_5X5_100,
    'aruco-5x5-250': Dictionary.ARUCO_5X5_250,
    'aruco-5x5-1000': Dictionary.ARUCO_5X5_1000,
    'aruco-6x6-50': Dictionary.ARUCO_6X6_50,
    'aruco-6x6-100': Dictionary.ARUCO_6X6_100,
    'aruco-6x6-250': Dictionary.ARUCO_6X6_250,
    'aruco-6x6-1000': Dictionary.ARUCO_6X6_1000,
    'aruco-7x7-50': Dictionary.ARUCO_7X7_50,
    'aruco-7x7-100': Dictionary.ARUCO_7X7_100,
    'aruco-7x7-250': Dictionary.ARUCO_7X7_250,
    'aruco-7x7-1000': Dictionary.ARUCO_7X7_1000,
    'aruco-mip-36h12': Dictionary.ARUCO_MIP_36H12,
    'april-16h5': Dictionary.APRIL_16H5,
    'april-25h9': Dictionary.APRIL_25H9,
    'april-36h10': Dictionary.APRIL_36H10,
    'april-36h11': Dictionary.APRIL_36H11,
}

_DICTIONARY_VARIANTS_TO_NAMES: dict[Dictionary, str] = dict(
    map(lambda entry: (entry[1], entry[0]), _DICTIONARY_NAMES_TO_VARIANTS.items())
)


class RigidModel:
    square_size: float
    coordinates: FloatArray2

    def __init__(self, square_size: float, depth: float) -> None:
        self.square_size = square_size

        self.coordinates = np.array(
            [
                [-square_size / 2.0, square_size / 2.0, depth],
                [square_size / 2.0, square_size / 2.0, depth],
                [square_size / 2.0, -square_size / 2.0, depth],
                [-square_size / 2.0, -square_size / 2.0, depth],
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class Result:
    corners: FloatArray3
    ids: IntArray1
    transformations: list[transformation.ProjectiveTransformation | None]

    # TODO: Implement custom drawing procedure
    def visualize(
        self,
        frame: Frame,
        frame_properties: Properties,
        configuration: Any,  # TODO: Add hint
    ) -> Frame:
        draw_boxes = configuration.marker_draw_bounding_boxes
        draw_ids = configuration.marker_draw_ids
        draw_axes = configuration.marker_draw_axes
        draw_angles = configuration.marker_draw_angles

        if draw_boxes:
            color = configuration.marker_bounding_box_color
            opencv_aruco.drawDetectedMarkers(frame, list(self.corners), self.ids, color)

        if draw_ids:
            ...  # TODO: implement with custom procedure

        calibration = frame_properties.calibration
        intrinsics = calibration.intrinsics
        distortion = calibration.distortion

        if draw_axes:
            length = configuration.marker_axis_length
            thickness = configuration.marker_axis_thickness

            for transformation in self.transformations:
                if transformation is None:
                    continue

                opencv.drawFrameAxes(
                    frame,
                    intrinsics,
                    distortion,
                    transformation.rotation,
                    transformation.translation,
                    length,
                    thickness,
                )

        if draw_angles:
            ...  # TODO: draw rotation angles around axes

        return frame


class Detector:
    dictionary: Dictionary
    raw_dictionary: opencv_aruco.Dictionary
    detector: opencv_aruco.ArucoDetector

    marker_size: float
    marker_local_coordinates: FloatArray2

    calibration: Calibration

    def __init__(
        self,
        *,
        model: RigidModel,
        dictionary: Dictionary,
        detector_parameters: opencv_aruco.DetectorParameters,
    ) -> None:
        self.dictionary = dictionary

        raw_dictionary = opencv_aruco.getPredefinedDictionary(dictionary.value)
        self.raw_dictionary = raw_dictionary

        self.detector = opencv_aruco.ArucoDetector(
            raw_dictionary,
            detector_parameters,
        )

        self.model = model

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

        marker_local_coordinates = self.model.coordinates

        results = [
            # interpretation: rotate the camera so the marker will be in the center
            opencv.solvePnP(
                marker_local_coordinates,
                marker_corners,
                intrinsics,
                distortion,
                useExtrinsicGuess=True,
                flags=opencv.SOLVEPNP_IPPE_SQUARE,
            )
            for marker_corners in corners
        ]

        transformations = [
            transformation.ProjectiveTransformation(
                opencv.Rodrigues(rotation)[0],  # type: ignore
                np.squeeze(translation),  # type: ignore
                calibration,
            )
            if success
            else None
            for success, rotation, translation in results
        ]

        return Result(batched_corners, ids, transformations)
