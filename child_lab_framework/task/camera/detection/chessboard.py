from dataclasses import dataclass

import cv2 as opencv
import numpy as np

from ....typing.array import FloatArray2, FloatArray3
from ....typing.video import Frame


@dataclass(frozen=True)
class Result:
    corners: FloatArray3


class Detector:
    box_size: float
    inner_corners_per_row: int
    inner_corners_per_column: int

    termination_criteria: opencv.typing.TermCriteria
    object_points_template: FloatArray2

    def __init__(
        self,
        *,
        inner_corners_per_row: int,
        inner_corners_per_column: int,
        termination_criteria: opencv.typing.TermCriteria = (
            opencv.TERM_CRITERIA_EPS + opencv.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        ),
    ) -> None:
        self.inner_corners_per_row = inner_corners_per_row
        self.inner_corners_per_column = inner_corners_per_column

        # Termination criteria for corner sub-pixel refinement
        self.termination_criteria = termination_criteria

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0). ...,(6,5,0)
        self.object_points_template = np.zeros(
            (self.inner_corners_per_column * self.inner_corners_per_row, 3), np.float32
        )
        self.object_points_template[:, :2] = np.mgrid[
            0 : self.inner_corners_per_row, 0 : self.inner_corners_per_column
        ].T.reshape(-1, 2)

    def predict(self, frame: Frame) -> Result | None:
        corners: list[FloatArray2]

        grayscale_frame = opencv.cvtColor(frame, opencv.COLOR_RGB2GRAY)
        found, corners = opencv.findChessboardCorners(
            grayscale_frame,
            (self.inner_corners_per_row, self.inner_corners_per_column),
            None,
        )

        if not found:
            return None

        corners = opencv.cornerSubPix(
            grayscale_frame, corners, (11, 11), (-1, -1), self.termination_criteria
        )

        return Result(corners)
