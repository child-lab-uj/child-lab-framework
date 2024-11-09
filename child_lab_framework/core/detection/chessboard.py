from dataclasses import dataclass

import cv2 as opencv
import numpy as np

from ...task import visualization
from ...typing.array import FloatArray2, FloatArray3
from ...typing.video import Frame
from .. import video


@dataclass(frozen=True)
class Properties:
    square_size: float
    inner_corners_per_row: int
    inner_corners_per_column: int


@dataclass(frozen=True)
class Result:
    corners: FloatArray3
    properties: Properties  # TODO: delete this field as soon as custom drawing procedure is implemented

    def visualize(
        self,
        frame: Frame,
        frame_properties: video.Properties,
        configuration: visualization.Configuration,
    ) -> Frame:
        if not configuration.chessboard_draw_corners:
            return frame

        properties = self.properties

        pattern_shape = (
            properties.inner_corners_per_row,
            properties.inner_corners_per_column,
        )

        # TODO: implement custom drawing
        opencv.drawChessboardCorners(frame, pattern_shape, self.corners, True)

        return frame


class Detector:
    properties: Properties
    square_size: float
    inner_corners_per_row: int
    inner_corners_per_column: int

    termination_criteria: opencv.typing.TermCriteria
    object_points_template: FloatArray2

    def __init__(
        self,
        properties: Properties,
        *,
        termination_criteria: opencv.typing.TermCriteria = (
            opencv.TERM_CRITERIA_EPS + opencv.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        ),
    ) -> None:
        self.properties = properties
        self.square_size = properties.square_size
        self.inner_corners_per_row = properties.inner_corners_per_row
        self.inner_corners_per_column = properties.inner_corners_per_column

        # Termination criteria for corner sub-pixel refinement
        self.termination_criteria = termination_criteria

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0). ...,(6,5,0)
        self.object_points_template = np.zeros(
            (properties.inner_corners_per_column * properties.inner_corners_per_row, 3),
            np.float32,
        )

        self.object_points_template[:, :2] = np.mgrid[
            0 : properties.inner_corners_per_row, 0 : properties.inner_corners_per_column
        ].T.reshape(-1, 2)

    # TODO: take grayscale frame as an argument to avoid conversion
    def predict(self, frame: Frame) -> Result | None:
        grayscale_frame = opencv.cvtColor(frame, opencv.COLOR_RGB2GRAY)
        properties = self.properties

        found, corners_dirty = opencv.findChessboardCorners(  # type: ignore
            grayscale_frame,
            (self.inner_corners_per_row, self.inner_corners_per_column),
            None,
        )

        if not found:
            return None

        corners: list[FloatArray2] = opencv.cornerSubPix(  # type: ignore
            grayscale_frame,
            corners_dirty,
            (11, 11),
            (-1, -1),
            self.termination_criteria,
        )

        return Result(np.stack(corners), properties)
