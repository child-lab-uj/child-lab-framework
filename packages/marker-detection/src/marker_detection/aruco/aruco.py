import typing
from dataclasses import dataclass
from enum import IntEnum
from typing import TypedDict

import cv2 as opencv
import cv2.aruco as opencv_aruco
import numpy
from jaxtyping import Float, Int
from video_io.frame import ArrayRgbFrame

from .draw import (
    draw_3d_dice_models,
    draw_marker_frame_axes,
    draw_marker_ids,
    draw_marker_masks,
)
from .geometry import (
    DistortionCoefficients,
    IntrinsicsMatrix,
    MarkerRigidModel,
    Transformation,
)

# TODO: Add serialization to `Dictionary`.


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


class VisualizationContext(TypedDict):
    intrinsics: IntrinsicsMatrix
    marker_draw_masks: bool
    marker_draw_ids: bool
    marker_draw_axes: bool
    marker_draw_angles: bool
    marker_draw_3d_dice_models: bool
    marker_mask_color: tuple[float, float, float, float]
    marker_axis_length: float
    marker_axis_thickness: int
    marker_rigid_model: MarkerRigidModel


@dataclass(frozen=True)
class Result:
    corners: Int[numpy.ndarray, 'n 4 2']
    ids: Int[numpy.ndarray, ' n']
    transformations: list[Transformation]

    def draw(
        self,
        frame: ArrayRgbFrame,
        context: VisualizationContext,
    ) -> ArrayRgbFrame:
        if len(self.ids) == 0:
            return frame

        if context['marker_draw_3d_dice_models']:
            draw_3d_dice_models(
                frame,
                self.transformations,
                context['intrinsics'],
                context['marker_rigid_model'],
            )

        if context['marker_draw_masks']:
            draw_marker_masks(frame, self.corners, context['marker_mask_color'])

        if context['marker_draw_axes']:
            draw_marker_frame_axes(
                frame,
                self.transformations,
                context['intrinsics'],
                context['marker_axis_length'],
                context['marker_axis_thickness'],
                context['marker_draw_angles'],
            )

        if context['marker_draw_ids']:
            draw_marker_ids(frame, self.ids, self.corners)

        return frame


class Detector:
    dictionary: Dictionary
    raw_dictionary: opencv_aruco.Dictionary
    detector: opencv_aruco.ArucoDetector

    marker_size: float
    marker_local_coordinates: Float[numpy.ndarray, '3 4']

    intrinsics: IntrinsicsMatrix
    distortion: DistortionCoefficients

    def __init__(
        self,
        model: MarkerRigidModel,
        dictionary: Dictionary,
        detector_parameters: opencv_aruco.DetectorParameters,
        intrinsics: IntrinsicsMatrix,
        distortion: DistortionCoefficients,
    ) -> None:
        self.dictionary = dictionary

        raw_dictionary = opencv_aruco.getPredefinedDictionary(dictionary.value)
        self.raw_dictionary = raw_dictionary

        self.detector = opencv_aruco.ArucoDetector(
            raw_dictionary,
            detector_parameters,
        )

        self.model = model
        self.intrinsics = intrinsics
        self.distortion = distortion

    def predict(self, frame: ArrayRgbFrame) -> Result | None:
        corners_dirty, ids_dirty, _rejected = self.detector.detectMarkers(frame)

        corners = typing.cast(list[Float[numpy.ndarray, 'n 4']], corners_dirty)
        ids = typing.cast(Int[numpy.ndarray, ' n'], ids_dirty)

        if len(corners) == 0 or len(ids) == 0:
            return None

        intrinsics = self.intrinsics
        distortion = self.distortion

        marker_rigid_coordinates = self.model.coordinates

        results = [
            opencv.solvePnP(
                marker_rigid_coordinates,
                camera_coordinates,
                intrinsics,
                distortion,
                useExtrinsicGuess=True,
                flags=opencv.SOLVEPNP_IPPE_SQUARE,
            )
            for camera_coordinates in corners
        ]

        transformations = [
            Transformation(opencv.Rodrigues(rotation)[0], numpy.squeeze(translation))
            for success, rotation, translation in results
            if success
        ]

        return Result(numpy.stack(corners), ids.reshape(-1), transformations)
