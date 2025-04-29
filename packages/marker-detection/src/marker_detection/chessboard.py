import math
import typing
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import ClassVar, TypedDict

import cv2 as opencv
import numpy
from jaxtyping import Float
from video_io import annotation
from video_io.frame import ArrayGrayFrame, ArrayRgbFrame

# +------------------------------------------------------------------------------------+
# | Algorithms and parts of code adopted from:                                         |
# | Repository: https://github.com/ros-perception/image_pipeline                       |
# | Files:                                                                             |
# |   - `image_pipeline/camera_calibration/src/camera_calibration/calibrator.py`       |
# |   - `image_pipeline/camera_calibration/src/camera_calibration/mono_calibrator.py`  |
# | Commit: 722ca08b98f37b7b148d429753da133ff1e2c7cf                                   |
# +------------------------------------------------------------------------------------+


type Corners = Float[numpy.ndarray, 'n 1 2']
type Corner = Float[numpy.ndarray, '2']


@dataclass(slots=True)
class BoardProperties:
    """
    Physical properties of the chessboard.
    """

    square_size: float
    inner_rows: int
    inner_columns: int

    @property
    def rigid_model(self) -> Float[numpy.ndarray, 'n 4']:
        """
        3D coordinates of the inner chessboard corners.
        """

        inner_rows = self.inner_rows
        inner_columns = self.inner_columns

        # How they do it in ros_image_pipeline:
        # total_points = inner_rows * inner_columns
        # object_points = numpy.zeros((total_points, 1, 3), numpy.float32)

        # for j in range(total_points):
        #     object_points[j, 0, 0] = j // inner_columns
        #     object_points[j, 0, 1] = j % inner_columns
        #     object_points[j, 0, 2] = 0
        #     object_points[j, 0, :] *= self.square_size

        model = numpy.zeros(
            (inner_rows * inner_columns, 3),
            numpy.float32,
        )

        corner_grid = numpy.mgrid[0:inner_columns, 0:inner_rows]

        # Flip the grid to have a "natural" orientation of axes and start from lower_left = (0, 0)
        model[:, :2] = numpy.flipud(corner_grid.T).reshape(-1, 2) * self.square_size

        return model


@dataclass(init=False, slots=True)
class AggregatedDetectionDetails:
    """
    Representation of `DetectionProperties` gathered from multiple detections,
    used to describe the heuristic qualities of the whole detection process.
    """

    __data: Float[numpy.ndarray, 'n 5']

    __AREA_WEIGHT: ClassVar[float] = 0.0
    __SKEW_WEIGHT: ClassVar[float] = 2.0
    __X_OFFSET_WEIGHT: ClassVar[float] = 1.4
    __Y_OFFSET_WEIGHT: ClassVar[float] = 1.4
    __PERSPECTIVE_OFFSET_WEIGHT: ClassVar[float] = 2.5

    __PROGRESS_WEIGHTS: ClassVar[Float[numpy.ndarray, 'n 5']] = numpy.array(
        (
            __AREA_WEIGHT,
            __SKEW_WEIGHT,
            __X_OFFSET_WEIGHT,
            __Y_OFFSET_WEIGHT,
            __PERSPECTIVE_OFFSET_WEIGHT,
        )
    )

    def __init__(self, results: Sequence['DetectionDetails']) -> None:
        n = len(results)

        data = numpy.empty((n, 5), dtype=numpy.float32)

        for i, item in enumerate(results):
            data[i, 0] = item.area
            data[i, 1] = item.skew
            data[i, 2] = item.x_offset
            data[i, 3] = item.y_offset
            data[i, 4] = item.perspective_offset

        self.__data = data

    def mean(self) -> 'DetectionDetails':
        """
        Compute the mean properties and store them as a `DetectionProperties`.
        """

        # use .tolist to convert numpy.float32 to Python's float
        return DetectionDetails(*numpy.mean(self.__data, axis=0).flatten().tolist())

    def progress(self) -> 'DetectionDetails':
        """
        Estimate the overall progress of the calibration
        based on the ranges of parameters.
        """

        data = self.__data

        max = numpy.max(data, axis=0).flatten()
        min = numpy.min(data, axis=0).flatten()

        # From ROS:
        min[3] = 0.0
        min[4] = 0.0

        value_range: Float[numpy.ndarray, '5'] = max - min
        progress = self.__PROGRESS_WEIGHTS * value_range

        # use .tolist to convert numpy.float32 to Python's float
        return DetectionDetails(*progress.tolist())


@dataclass(slots=True)
class DetectionDetails:
    """
    Heuristic properties of the chessboard detection,
    describing its orientation relative to the camera.

    Attributes
    ---
    area: float
        Area of the projection of the board.

    skew: float
        Angle between the pair of adjacent edges.

    x_offset: float
        Displacement along the X-axis.

    y_offset: float
        Displacement along the Y-axis.

    perspective_offset: float
        Displacement in the camera's perspective, dependent on the size of the projection.
    """

    area: float
    skew: float
    x_offset: float
    y_offset: float
    perspective_offset: float

    def distance(self, other: 'DetectionDetails') -> float:
        """
        Compute the Manhattan distance between two observations
        in the space of values stored in `DetectionProperties`.
        """

        return (
            abs(self.skew - other.skew)
            + abs(self.x_offset - other.x_offset)
            + abs(self.y_offset - other.y_offset)
            + abs(self.perspective_offset - other.perspective_offset)
        )


class VisualizationContext(TypedDict):
    chessboard_draw_corners: bool


@dataclass(slots=True)
class Result:
    """
    Description of the chessboard detection.

    Attributes
    ---
    corners: Corners
        `n_detection x 1 x 2` array containing the positions of the detected inner corners.

    board_properties: BoardProperties
        Properties of the detected board.

    details: DetectionDetails
        Additional information about the detection in relation to the camera.
    """

    corners: Corners
    board_properties: BoardProperties
    details: DetectionDetails

    def __init__(
        self,
        corners: Corners,
        board_properties: BoardProperties,
        frame_width: int,
        frame_height: int,
    ) -> None:
        self.corners = corners
        self.board_properties = board_properties
        self.details = self.__detection_details(frame_width, frame_height)

    def __detection_details(
        self,
        frame_width: int,
        frame_height: int,
    ) -> DetectionDetails:
        """
        Calculate the properties describing the heuristic quality of the result.
        """
        skew = self.skew
        area = self.area

        border = math.sqrt(area)

        x_mean = float(numpy.mean(self.corners[..., 0]))
        y_mean = float(numpy.mean(self.corners[..., 1]))

        x_offset = min(1.0, max(0.0, (x_mean - 0.5 * border) / (frame_width - border)))
        y_offset = min(1.0, max(0.0, (y_mean - 0.5 * border) / (frame_height - border)))
        perspective_offset = area / float(frame_width * frame_height)

        return DetectionDetails(
            area,
            skew,
            x_offset,
            y_offset,
            perspective_offset,
        )

    @property
    def area(self) -> float:
        """
        Calculate the area of the board.
        """
        # Assumes the board is a convex quadrilateral.

        upper_left, upper_right, lower_left, lower_right = self.outer_corners

        diagonal_1_x, diagonal_1_y = lower_right - upper_left
        diagonal_2_x, diagonal_2_y = lower_left - upper_right

        return 0.5 * abs(float(diagonal_1_x * diagonal_2_y - diagonal_1_y * diagonal_2_x))

    @property
    def skew(self) -> float:
        """
        Calculate the skew of the board.
        """

        upper_left, upper_right, lower_left, _ = self.outer_corners

        vertical_line = upper_right - upper_left
        horizontal_line = lower_left - upper_left

        dot_product = numpy.dot(vertical_line, horizontal_line)
        norm_product = numpy.linalg.norm(vertical_line) * numpy.linalg.norm(
            horizontal_line
        )
        angle = float(numpy.arccos(dot_product / norm_product))

        return min(1.0, 2.0 * abs(numpy.pi / 2.0 - angle))

    @property
    def outer_corners(
        self,
    ) -> tuple[Corner, Corner, Corner, Corner]:
        """
        Outer corners of the board, in the following order:
        upper-left, upper-right, lower-left, lower-right.
        """

        rows = self.board_properties.inner_rows
        columns = self.board_properties.inner_columns

        corners = self.corners

        return (
            corners[0, 0],
            corners[columns - 1, 0],
            corners[(rows - 1) * columns, 0],
            corners[rows * columns - 1, 0],
        )

    def average_speed(self, previous: 'Result', time_delta: float) -> float:
        """
        Calculate the approximate average speed of the moving board,
        assuming that `previous` was the last known displacement `time_delta` seconds ago.
        """

        return (
            float(numpy.average(numpy.linalg.norm(self.corners - previous.corners)))
            / time_delta
        )

    def draw(
        self,
        frame: ArrayRgbFrame,
        context: VisualizationContext,
    ) -> ArrayRgbFrame:
        """
        Draw the inner corners of the chessboard with colors depending on their order.
        """

        if not context['chessboard_draw_corners']:
            return frame

        properties = self.board_properties

        pattern_shape = (
            properties.inner_columns,
            properties.inner_rows,
        )

        opencv.drawChessboardCorners(frame, pattern_shape, self.corners, True)

        upper_left, upper_right, lower_left, lower_right = self.outer_corners

        area_vertices = numpy.stack(
            (
                upper_left.astype(numpy.int32),
                upper_right.astype(numpy.int32),
                lower_right.astype(numpy.int32),
                lower_left.astype(numpy.int32),
            )
        )

        distance = self.details.perspective_offset

        annotation.draw_polygon_with_description(
            frame,
            area_vertices,
            f'{distance = :.2f}',
            area_opacity=0.15,
            font_scale=1.5,
            font_thickness=2,
            box_opacity=0.30,
        )

        annotation.draw_point_with_description(
            frame,
            upper_left.astype(int).tolist(),
            'upper left',
            text_location='above',
        )

        annotation.draw_point_with_description(
            frame,
            upper_right.astype(int).tolist(),
            'upper right',
            text_location='above',
        )

        annotation.draw_point_with_description(
            frame,
            lower_left.astype(int).tolist(),
            'lower left',
        )

        annotation.draw_point_with_description(
            frame,
            lower_right.astype(int).tolist(),
            'lower right',
        )

        for corner, rigid_point in zip(
            numpy.squeeze(self.corners),
            numpy.squeeze(self.board_properties.rigid_model),
        ):
            x, y = corner.astype(int)
            rx, ry, _ = rigid_point
            annotation.draw_point_with_description(
                frame,
                (x, y),
                f'({rx:.2f}, {ry:.2f})',
                font_scale=0.25,
            )

        return frame


@dataclass(slots=True)
class Detector:
    board_properties: BoardProperties
    termination_criteria: opencv.typing.TermCriteria = field(
        default=(
            opencv.TERM_CRITERIA_EPS + opencv.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
    )

    # TODO: take grayscale frame as an argument to avoid conversion
    def predict(self, frame: ArrayRgbFrame) -> Result | None:
        """
        Detect the board in the frame.
        """

        gray_frame = typing.cast(
            ArrayGrayFrame,
            opencv.cvtColor(frame, opencv.COLOR_RGB2GRAY),
        )

        downscaled_gray_frame, area_scale = self.__downscale(gray_frame)

        original_height, original_width, _ = frame.shape
        height, width = downscaled_gray_frame.shape

        board_properties = self.board_properties

        found, corners_dirty = opencv.findChessboardCorners(
            downscaled_gray_frame,
            (board_properties.inner_columns, board_properties.inner_rows),
            None,
            opencv.CALIB_CB_ADAPTIVE_THRESH
            | opencv.CALIB_CB_NORMALIZE_IMAGE
            | opencv.CALIB_CB_FAST_CHECK,
        )

        if not found:
            return None

        corners = typing.cast(Corners, corners_dirty)

        if self.__is_close_to_edge(corners, 10.0, width, height):
            return None

        corners = self.__normalize_orientation(corners, board_properties)
        corners = self.__refine(downscaled_gray_frame, corners)
        corners = self.__upscale(
            gray_frame,
            corners,
            original_width / width,
            original_height / height,
            area_scale,
        )

        return Result(
            corners,
            board_properties,
            original_width,
            original_height,
        )

    def __downscale(self, frame: ArrayGrayFrame) -> tuple[ArrayGrayFrame, float]:
        """
        Resize the frame to a size comparable to 640 x 480 px, preserving the aspect ratio.

        Returns
        ---
        result: tuple[Frame, float]
            The resized frame and the scale used.
        """

        height, width = frame.shape
        scale = math.sqrt(307_200.0 / (height * width))

        new_size = (
            int(math.ceil(width * scale)),
            int(math.ceil(height * scale)),
        )

        resized_frame = opencv.resize(frame, new_size)

        return typing.cast(ArrayGrayFrame, resized_frame), scale

    def __upscale(
        self,
        original_frame: ArrayGrayFrame,
        corners: Corners,
        x_scale: float,
        y_scale: float,
        area_scale: float,
    ) -> Corners:
        """
        Upscale the corner coordinates with `x_scale` and `y_scale` along respective axes
        and perform sub-pixel correction.

        Parameters
        ---
        original_frame: ArrayGrayFrame
            Original, non-scaled gray-scale frame.

        corners: Corners
            Corners of the board detected on the downsampled frame.

        x_scale: float
            Scale for the corner coordinates along X-axis.

        y_scale: float
        Scale for the corner coordinates along Y-axis.

        area_scale: float
            Scale obtained from `_downscale`.

        termination_criteria: opencv.typing.TermCriteria
            Termination criteria from the sub-pixel refinement.
        """

        corners[..., 0] *= x_scale
        corners[..., 1] *= y_scale

        radius = int(math.ceil(area_scale))

        corners_dirty = opencv.cornerSubPix(
            original_frame,
            corners,
            (radius, radius),
            (-1, -1),
            self.termination_criteria,
        )

        return corners_dirty

    def __is_close_to_edge(
        self,
        corners: Corners,
        margin: float,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        """
        Check if the detected board is closer to the edges of the frame than `margin`.
        """

        xs_lower = corners[:, 0, 0] <= margin
        ys_lower = corners[:, 0, 1] <= margin
        xs_greater = corners[:, 0, 0] >= (float(frame_width) - margin)
        ys_greater = corners[:, 0, 1] >= (float(frame_height) - margin)

        return bool(numpy.any(xs_lower | ys_lower | xs_greater | ys_greater))

    def __normalize_orientation(
        self,
        corners: Corners,
        board: BoardProperties,
    ) -> Corners:
        """
        Sort the `corners` increasingly and reshape them to `n_detections x board_width x board_height`.
        """

        rows = board.inner_rows
        columns = board.inner_columns

        if rows != columns:
            if corners[0, 0, 1] > corners[-1, 0, 1]:
                return numpy.ascontiguousarray(numpy.flipud(corners))

            return corners

        direction_indicator: list[bool] = numpy.squeeze(
            (corners[-1] - corners[0]) >= 0.0
        ).tolist()

        match direction_indicator:
            case [True, True]:
                return corners

            case [True, False]:
                corners_2d_grid = corners.reshape(rows, columns, 2)
                corners_rotated = numpy.rot90(corners_2d_grid).reshape(-1, 1, 2)
                return numpy.ascontiguousarray(corners_rotated)

            case [False, True]:
                corners_2d_grid = corners.reshape(rows, columns, 2)
                corners_rotated = numpy.rot90(corners_2d_grid, 3).reshape(-1, 1, 2)
                return numpy.ascontiguousarray(corners_rotated)

            case [False, False]:
                return numpy.ascontiguousarray(numpy.flipud(corners))

        assert False, 'unreachable'

    def __refine(
        self,
        frame: ArrayGrayFrame,
        corners: Corners,
    ) -> Corners:
        """
        Correct the corner displacement according to the mutual distance between them.
        """

        xs = corners[..., 0].reshape(1, -1)
        ys = corners[..., 1].reshape(1, -1)

        pairwise_distance = numpy.sqrt((xs - xs.T) ** 2 + (ys - ys.T) ** 2)
        numpy.fill_diagonal(pairwise_distance, numpy.inf)

        minimal_distance = numpy.min(pairwise_distance)
        radius = int(numpy.ceil(0.5 * minimal_distance))

        corners_dirty = opencv.cornerSubPix(
            frame,
            corners,
            (radius, radius),
            (-1, -1),
            self.termination_criteria,
        )

        return corners_dirty
