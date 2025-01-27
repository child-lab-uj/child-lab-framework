import asyncio
import typing
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat, starmap

import cv2
import numpy as np
from icecream import ic

from ...core import transformation
from ...core.stream import InvalidArgumentException
from ...core.transformation import Transformation
from ...core.video import Properties
from ...typing.array import BoolArray1, FloatArray1, FloatArray2, IntArray1
from ...typing.stream import Fiber
from ...typing.video import Frame
from .. import pose, visualization
from . import baseline, gaze

type Input = tuple[
    list[pose.Result | None] | None,
    list[FloatArray2 | None] | None,
    list[gaze.Result | None] | None,
    list[gaze.Result | None] | None,
    list[Transformation | None] | None,
    list[Transformation | None] | None,
]


@dataclass(frozen=True, slots=True)
class Result:
    centers: FloatArray2
    directions: FloatArray2
    was_corrected: BoolArray1

    center_depths: FloatArray1 | None = None

    _baseline_gaze: visualization.Visualizable | None = None
    """For visualization purposes only."""

    def visualize(
        self,
        frame: Frame,
        frame_properties: Properties,
        configuration: visualization.Configuration,
    ) -> Frame:
        if configuration.gaze_draw_baseline and self._baseline_gaze is not None:
            self._baseline_gaze.visualize(frame, frame_properties, configuration)

        directions = float(configuration.gaze_line_length) * self.directions

        starts = self.centers
        ends = starts + directions

        start: IntArray1
        end: IntArray1

        color = configuration.gaze_line_color
        thickness = configuration.gaze_line_thickness

        for start, end in zip(
            starts.astype(np.int32),
            ends.astype(np.int32),
        ):
            cv2.line(
                frame,
                typing.cast(cv2.typing.Point, start),
                typing.cast(cv2.typing.Point, end),
                color,
                thickness,
            )

        return frame


class Estimator:
    executor: ThreadPoolExecutor | None

    transformation_buffer: transformation.Buffer[str]

    ceiling_properties: Properties
    window_left_properties: Properties
    window_right_properties: Properties

    def __init__(
        self,
        transformation_buffer: transformation.Buffer[str],
        ceiling_properties: Properties,
        window_left_properties: Properties,
        window_right_properties: Properties,
        *,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.executor = executor
        self.transformation_buffer = transformation_buffer
        self.ceiling_properties = ceiling_properties
        self.window_left_properties = window_left_properties
        self.window_right_properties = window_right_properties

    def predict(
        self,
        ceiling_pose: pose.Result,
        ceiling_depth: FloatArray2 | None,
        window_left_gaze: gaze.Result3d | None,
        window_right_gaze: gaze.Result3d | None,
        window_left_to_ceiling: Transformation | None,
        window_right_to_ceiling: Transformation | None,
    ) -> Result:
        buffer = self.transformation_buffer

        ceiling_name = self.ceiling_properties.name
        window_left_name = self.window_left_properties.name
        window_right_name = self.window_right_properties.name

        if window_left_to_ceiling is None:
            window_left_to_ceiling = buffer[window_left_name, ceiling_name]

        if window_right_to_ceiling is None:
            window_right_to_ceiling = buffer[window_right_name, ceiling_name]

        baseline_gaze = (
            ceiling_depth is not None
            and baseline.head.estimate(
                ceiling_pose, ceiling_depth, 0.8, child_quantile=0.9
            )
            or baseline.keypoint.estimate(
                ceiling_pose,
                face_keypoint_threshold=0.4,
            )
        )

        centers = baseline_gaze.centers.copy()
        directions = baseline_gaze.directions.copy()

        for i, actor in enumerate(ceiling_pose.actors):
            if actor == pose.Actor.CHILD:
                directions[i, :] *= -1.0

        correct_from_left = False
        correct_from_right = False

        # if len(directions) > 1 and window_left_gaze is not None:
        #     left_correction_angle = (
        #         window_left_gaze.horizontal_offset_angles[0] - np.pi / 2.0
        #     )

        #     if not np.isnan(left_correction_angle):
        #         sin = np.sin(left_correction_angle)
        #         cos = np.cos(left_correction_angle)
        #         rotation = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)

        #         directions[1, :] = rotation @ directions[1, :]

        # if window_right_gaze is not None:
        #     right_correction_angle = -(
        #         window_right_gaze.horizontal_offset_angles[0] - np.pi / 2.0
        #     )

        #     if not np.isnan(right_correction_angle):
        #         sin = np.sin(right_correction_angle)
        #         cos = np.cos(right_correction_angle)
        #         rotation = np.array([[cos, -sin], [sin, cos]], dtype=np.float32)

        #         directions[0, :] = rotation @ directions[0, :]

        if ceiling_depth is not None:
            center_columns, center_rows = centers.astype(int).T
            center_depths = ceiling_depth[center_rows, center_columns]
        else:
            center_depths = None

        # return Result(
        #     centers,
        #     directions,
        #     np.array(
        #         [correct_from_right, correct_from_left]
        #     ),  # assumption about two actors...
        #     center_depths=center_depths,
        #     _baseline_gaze=baseline_gaze,
        # )

        ceiling_calibration = self.ceiling_properties.calibration

        # slicing gaze direction arrays is a heuristic workaround for a lack of exact actor matching between cameras
        # assuming there are only two actors in the world.
        # gaze detected in the left camera => it belongs the actor on the right

        if window_left_gaze is not None and window_left_to_ceiling is not None:
            correct_from_left = True

            projected_gaze = window_left_gaze.transform(window_left_to_ceiling).project(
                ceiling_calibration
            )

            projected_directions = np.squeeze(np.mean(projected_gaze.directions, axis=1))
            projected_centres = np.mean(projected_gaze.eyes, axis=1).reshape(-1, 2)
            ic(projected_centres)

            centers[-1, ...] = projected_centres[-1, ...]
            directions[-1, ...] = projected_directions[-1, ...]

        if window_right_gaze is not None and window_right_to_ceiling is not None:
            correct_from_right = True

            projected_gaze = window_right_gaze.transform(window_right_to_ceiling).project(
                ceiling_calibration
            )

            projected_directions = np.squeeze(np.mean(projected_gaze.directions, axis=1))
            projected_centres = np.mean(projected_gaze.eyes, axis=1).reshape(-1, 2)
            ic(projected_centres)

            centers[0, ...] = projected_centres[0, ...]
            directions[0, ...] = projected_directions[0, ...]

        return Result(
            centers,
            directions,
            np.array(
                [correct_from_right, correct_from_left]
            ),  # assumption about two actors...
            center_depths,
            baseline_gaze,
        )

    def predict_batch(
        self,
        ceiling_poses: list[pose.Result],
        ceiling_depths: list[FloatArray2] | None,
        window_left_gazes: list[gaze.Result3d] | None,
        window_right_gazes: list[gaze.Result3d] | None,
        window_left_to_ceiling: list[Transformation] | None,
        window_right_to_ceiling: list[Transformation] | None,
    ) -> list[Result | None]:
        return list(
            starmap(
                self.predict,
                zip(
                    ceiling_poses,
                    ceiling_depths or repeat(None),
                    window_left_gazes or repeat(None),
                    window_right_gazes or repeat(None),
                    window_left_to_ceiling or repeat(None),
                    window_right_to_ceiling or repeat(None),
                ),
            )
        )

    def __predict_safe(
        self,
        ceiling_pose: pose.Result | None,
        ceiling_depth: FloatArray2 | None,
        window_left_gaze: gaze.Result3d | None,
        window_right_gaze: gaze.Result3d | None,
        window_left_to_ceiling: Transformation | None,
        window_right_to_ceiling: Transformation | None,
    ) -> Result | None:
        if ceiling_pose is None:
            return None

        return self.predict(
            ceiling_pose,
            ceiling_depth,
            window_left_gaze,
            window_right_gaze,
            window_left_to_ceiling,
            window_right_to_ceiling,
        )

    # NOTE: heuristic idea: actors seen from right and left are in reversed lexicographic order
    async def stream(self) -> Fiber[Input | None, list[Result | None] | None]:
        executor = self.executor
        if executor is None:
            raise RuntimeError(
                'Processing in the stream mode requires the Estimator to have an executor. Please pass an "executor" argument to the estimator constructor'
            )

        loop = asyncio.get_running_loop()

        results: list[Result | None] | None = None

        while True:
            match (yield results):
                case (
                    list(ceiling_pose),
                    ceiling_depth,
                    window_left_gaze,
                    window_right_gaze,
                    window_left_to_ceiling,
                    window_right_to_ceiling,
                ):
                    results = await loop.run_in_executor(
                        executor,
                        lambda: list(
                            starmap(
                                self.__predict_safe,
                                zip(
                                    ceiling_pose,
                                    ceiling_depth or repeat(None),
                                    window_left_gaze or repeat(None),
                                    window_right_gaze or repeat(None),
                                    window_left_to_ceiling or repeat(None),
                                    window_right_to_ceiling or repeat(None),
                                ),
                            )
                        ),
                    )

                case None, _, _, _, _:
                    results = None

                case _:
                    raise InvalidArgumentException()
