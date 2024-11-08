import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat, starmap

import numpy as np

from ...core.sequence import imputed_with_reference_inplace
from ...core.stream import InvalidArgumentException
from ...core.transformation import ProjectiveTransformation
from ...core.video import Properties
from ...logging import Logger
from ...typing.array import BoolArray1, FloatArray2
from ...typing.stream import Fiber
from .. import face, pose
from . import ceiling_baseline, gaze

type Input = tuple[
    list[pose.Result | None] | None,
    list[pose.Result | None] | None,
    list[pose.Result | None] | None,
    list[face.Result | None] | None,
    list[face.Result | None] | None,
    list[ProjectiveTransformation | None] | None,
    list[ProjectiveTransformation | None] | None,
]


@dataclass(frozen=True)
class Result:
    centres: FloatArray2
    directions: FloatArray2
    was_corrected: BoolArray1


class Estimator:
    BASELINE_WEIGHT: float = 0.0
    COLLECTIVE_CORRECTION_WEIGHT: float = 1.0 - BASELINE_WEIGHT
    TEMPORARY_RESCALE: float = 1000.0  # for numerical stability during projection

    ceiling_properties: Properties
    window_left_properties: Properties
    window_right_properties: Properties

    executor: ThreadPoolExecutor

    def __init__(
        self,
        executor: ThreadPoolExecutor,
        ceiling_properties: Properties,
        window_left_properties: Properties,
        window_right_properties: Properties,
    ) -> None:
        self.executor = executor
        self.ceiling_properties = ceiling_properties
        self.window_left_properties = window_left_properties
        self.window_right_properties = window_right_properties

    def predict(
        self,
        ceiling_pose: pose.Result,
        window_left_gaze: gaze.Result | None,
        window_right_gaze: gaze.Result | None,
        window_left_to_ceiling: ProjectiveTransformation | None,
        window_right_to_ceiling: ProjectiveTransformation | None,
    ) -> Result:
        centres, directions = ceiling_baseline.estimate(
            ceiling_pose,
            face_keypoint_threshold=0.4,
        )

        correct_from_left = (
            window_left_gaze is not None and window_left_to_ceiling is not None
        )

        correct_from_right = (
            window_right_gaze is not None and window_right_to_ceiling is not None
        )

        correction_count = int(correct_from_left) + int(correct_from_right)

        if correction_count == 0:
            Logger.info('Skipped correction')
            return Result(
                centres,
                directions,
                np.array([False for _ in range(len(centres))]),
            )

        correction_weight = self.COLLECTIVE_CORRECTION_WEIGHT / float(correction_count)
        baseline_weight = self.BASELINE_WEIGHT
        temporary_rescale = self.TEMPORARY_RESCALE

        # slicing gaze direction arrays is a heuristic workaround for a lack of exact actor matching between cameras
        # assuming there are only two actors in the world.
        # gaze detected in the left camera => it belongs the actor on the right

        # cannot reuse `correct_from_left` value because the type checker gets confused about not-None values
        if window_left_gaze is not None and window_left_to_ceiling is not None:
            directions_simplified = (
                np.squeeze(np.mean(window_left_gaze.directions, axis=1))
                * temporary_rescale
            )

            directions_projected = (
                window_left_to_ceiling.project(directions_simplified)
                / temporary_rescale
                * correction_weight
            )

            directions[-1, ...] *= baseline_weight
            directions[-1, ...] += directions_projected[-1, ...]

        if window_right_gaze is not None and window_right_to_ceiling is not None:
            directions_simplified = np.squeeze(
                np.mean(window_right_gaze.directions, axis=1) * temporary_rescale
            )

            directions_projected = (
                window_right_to_ceiling.project(directions_simplified)
                / temporary_rescale
                * correction_weight
            )

            directions[0, ...] *= baseline_weight
            directions[0, ...] += directions_projected[0, ...]

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
        window_left_gazes: list[gaze.Result] | None,
        window_right_gazes: list[gaze.Result] | None,
        window_left_to_ceiling: list[ProjectiveTransformation] | None,
        window_right_to_ceiling: list[ProjectiveTransformation] | None,
    ) -> list[Result] | None:
        return imputed_with_reference_inplace(
            list(
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
        )

    def __predict_safe(
        self,
        ceiling_pose: pose.Result | None,
        window_left_gaze: gaze.Result | None,
        window_right_gaze: gaze.Result | None,
        window_left_to_ceiling: ProjectiveTransformation | None,
        window_right_to_ceiling: ProjectiveTransformation | None,
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
