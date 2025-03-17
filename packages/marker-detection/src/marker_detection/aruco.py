import typing
from dataclasses import dataclass
from enum import IntEnum
from typing import TypedDict

import cv2 as opencv
import cv2.aruco as opencv_aruco
import numpy
import scipy.spatial.transform as scipy_transform
from jaxtyping import Float, Int
from video_io import annotation
from video_io.frame import ArrayRgbFrame

# type Array = NDArray[numpy.float32]

type IntrinsicsMatrix = Float[numpy.ndarray, '3 3']
type DistortionCoefficients = Float[numpy.ndarray, '5']


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


class RigidModel:
    square_size: float
    coordinates: Float[numpy.ndarray, '3 4']

    def __init__(self, square_size: float, depth: float) -> None:
        self.square_size = square_size

        self.coordinates = numpy.array(
            [
                [-square_size / 2.0, square_size / 2.0, depth],
                [square_size / 2.0, square_size / 2.0, depth],
                [square_size / 2.0, -square_size / 2.0, depth],
                [-square_size / 2.0, -square_size / 2.0, depth],
            ],
            dtype=numpy.float32,
        )


class VisualizationContext(TypedDict):
    intrinsics: IntrinsicsMatrix
    marker_draw_masks: bool
    marker_draw_ids: bool
    marker_draw_axes: bool
    marker_draw_angles: bool
    marker_mask_color: tuple[float, float, float, float]
    marker_axis_length: int
    marker_axis_thickness: int


@dataclass(frozen=True, slots=True)
class Transformation:
    rotation: Float[numpy.ndarray, '3 3']
    translation: Float[numpy.ndarray, '3']

    def euler_angles(self) -> Float[numpy.ndarray, '3']:
        return (  # type: ignore[no-any-return]
            scipy_transform.Rotation.from_matrix(self.rotation).as_euler(
                'xyz',
                degrees=False,
            )
        )


@dataclass(frozen=True)
class Result:
    corners: Int[numpy.ndarray, 'n 4']
    ids: Int[numpy.ndarray, ' n']
    transformations: list[Transformation]

    def draw(
        self,
        frame: ArrayRgbFrame,
        context: VisualizationContext,
    ) -> ArrayRgbFrame:
        if len(self.ids) == 0:
            return frame

        draw_boxes = context['marker_draw_masks']
        draw_ids = context['marker_draw_ids']
        draw_axes = context['marker_draw_axes']
        draw_angles = context['marker_draw_angles']

        # TODO: Get rid of nested parts. It's better to have the same loop a few times.

        if draw_boxes:
            r, g, b, _ = context['marker_mask_color']
            color = (int(r), int(g), int(b))

            for marker_corners_raw in self.corners:
                marker_corners = marker_corners_raw.reshape(-1, 2).astype(numpy.int32)

                annotation.draw_filled_polygon_with_opacity(
                    frame,
                    marker_corners,
                    color=color,
                    opacity=0.5,
                )

                corner_pixels: list[tuple[int, int]] = list(map(tuple, marker_corners))
                upper_left, upper_right, lower_right, lower_left = corner_pixels

                annotation.draw_point_with_description(
                    frame,
                    upper_left,
                    'upper_left',
                    point_radius=1,
                    font_scale=0.4,
                    text_location='above',
                )
                annotation.draw_point_with_description(
                    frame,
                    upper_right,
                    'upper_right',
                    point_radius=1,
                    font_scale=0.4,
                    text_location='above',
                )
                annotation.draw_point_with_description(
                    frame,
                    lower_right,
                    'lower_right',
                    point_radius=1,
                    font_scale=0.4,
                )
                annotation.draw_point_with_description(
                    frame,
                    lower_left,
                    'lower_left',
                    point_radius=1,
                    font_scale=0.4,
                )

        if draw_axes:
            length = context['marker_axis_length']
            thickness = context['marker_axis_thickness']

            x_color = (255, 0, 0)
            y_color = (0, 255, 0)
            z_color = (0, 0, 255)

            intrinsics = context['intrinsics']

            basis_points = numpy.array(
                [
                    [0.0, 0.0, 0.0],
                    [length, 0.0, 0.0],
                    [0.0, length, 0.0],
                    [0.0, 0.0, length],
                ],
                dtype=numpy.float32,
            ).T

            for transformation in self.transformations:
                rotation = transformation.rotation
                translation = transformation.translation.reshape(-1, 1)

                transformed_basis = rotation @ basis_points + translation
                transformed_basis /= transformed_basis[2]
                projected_basis = intrinsics @ transformed_basis

                projected_basis_pixel_coordinates: list[tuple[int, int]] = (
                    projected_basis.T[:, :2].astype(int).tolist()
                )
                o, x, y, z = projected_basis_pixel_coordinates

                opencv.line(frame, o, x, x_color, thickness)
                opencv.line(frame, o, y, y_color, thickness)
                opencv.line(frame, o, z, z_color, thickness)

                if draw_angles:
                    angles: list[float] = (
                        180.0 / numpy.pi * transformation.euler_angles()
                    ).tolist()

                    x_angle, y_angle, z_angle = angles

                    annotation.draw_point_with_description(
                        frame,
                        x,
                        f'x: {x_angle:.2f}',
                        font_scale=0.3,
                        point_radius=1,
                        point_color=x_color,
                    )
                    annotation.draw_point_with_description(
                        frame,
                        y,
                        f'y: {y_angle:.2f}',
                        font_scale=0.3,
                        point_radius=1,
                        point_color=y_color,
                    )
                    annotation.draw_point_with_description(
                        frame,
                        z,
                        f'z: {z_angle:.2f}',
                        font_scale=0.3,
                        point_radius=1,
                        point_color=z_color,
                    )

        if draw_ids:
            for id, marker_corners in zip(self.ids, self.corners):
                marker_corners = marker_corners.reshape(-1, 2)
                center: tuple[int, int] = (
                    numpy.mean(marker_corners, axis=0).astype(int).tolist()
                )

                annotation.draw_text_within_box(
                    frame,
                    f'marker {id}',
                    center,
                    font_scale=0.3,
                )

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
        model: RigidModel,
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
