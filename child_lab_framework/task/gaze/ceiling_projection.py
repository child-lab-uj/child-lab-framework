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
from ...typing.array import BoolArray1, FloatArray2, IntArray1
from ...typing.stream import Fiber
from ...typing.video import Frame
from .. import face, pose, visualization
from . import ceiling_baseline, gaze

type Input = tuple[
    list[pose.Result | None] | None,
    list[pose.Result | None] | None,
    list[pose.Result | None] | None,
    list[face.Result | None] | None,
    list[face.Result | None] | None,
    list[Transformation | None] | None,
    list[Transformation | None] | None,
]


@dataclass(frozen=True)
class Result:
    centres: FloatArray2
    directions: FloatArray2
    was_corrected: BoolArray1

    def visualize(
        self,
        frame: Frame,
        frame_properties: Properties,
        configuration: visualization.Configuration,
    ) -> Frame:
        starts = self.centres
        ends = starts + float(configuration.gaze_line_length) * self.directions

        start: IntArray1
        end: IntArray1

        basic_color = configuration.gaze_line_color
        baseline_color = configuration.gaze_baseline_line_color
        thickness = configuration.gaze_line_thickness

        for start, end, was_corrected in zip(
            starts.astype(np.int32),
            ends.astype(np.int32),
            self.was_corrected,
        ):
            color = basic_color if was_corrected else baseline_color

            cv2.circle(
                frame,
                typing.cast(cv2.typing.Point, start),
                3,
                (0.0, 0.0, 255.0, 1.0),
                1,
            )

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

        centres, directions = ceiling_baseline.estimate(
            ceiling_pose,
            face_keypoint_threshold=0.4,
        )

        correct_from_left = False
        correct_from_right = False

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

            centres[-1, ...] = projected_centres[-1, ...]
            directions[-1, ...] = projected_directions[-1, ...]

        if window_right_gaze is not None and window_right_to_ceiling is not None:
            correct_from_right = True

            projected_gaze = window_right_gaze.transform(window_right_to_ceiling).project(
                ceiling_calibration
            )

            projected_directions = np.squeeze(np.mean(projected_gaze.directions, axis=1))
            projected_centres = np.mean(projected_gaze.eyes, axis=1).reshape(-1, 2)
            ic(projected_centres)

            centres[0, ...] = projected_centres[0, ...]
            directions[0, ...] = projected_directions[0, ...]

        return Result(
            centres,
            directions,
            np.array(
                [correct_from_right, correct_from_left]
            ),  # assumption about two actors...
        )

    def predict_batch(
        self,
        ceiling_poses: list[pose.Result],
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
        window_left_gaze: gaze.Result3d | None,
        window_right_gaze: gaze.Result3d | None,
        window_left_to_ceiling: Transformation | None,
        window_right_to_ceiling: Transformation | None,
    ) -> Result | None:
        if ceiling_pose is None:
            return None

        return self.predict(
            ceiling_pose,
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
