from dataclasses import dataclass
from enum import IntEnum
import numpy as np
from collections.abc import Generator, Iterable
import typing
from typing import Literal
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ...core.stream import autostart
from ...core.video import Perspective, Properties
from ...core.algebra import normalized, orthogonal, Axis, rotation_matrix
from .. import pose, face
from ..pose.keypoint import YoloKeypoint
from ...typing.array import FloatArray1, FloatArray2, FloatArray3
from ...typing.stream import Fiber
from . import ceiling_baseline, side_correction


# Multi-camera gaze direction estimation without strict algebraic camera models:
# 1. estimate actor's skeleton on each frame in both cameras
#    (heuristic: adults' keypoints have higher variance, children have smaller bounding boxes)
# 2. compute a ceiling baseline vector (perpendicular to shoulder line in a ceiling camera)
# 3. detect actor's face on the other camera
# 4. compute the offset-baseline vector (normal to face, Wim's MediaPipe solution')
# 5. Rotate it to the celing camera's space and combine with the ceiling baseline


type Input = tuple[
    list[pose.Result | None] | None,
    list[pose.Result | None] | None,
    list[pose.Result | None] | None,
    list[face.Result | None] | None,
    list[face.Result | None] | None,
]


@dataclass
class Result:
    centres: FloatArray2
    versors: FloatArray2

    def iter(self) -> Iterable[tuple[FloatArray1, FloatArray1]]:
        return zip(self.centres, self.versors)


class Estimator:
    BASELINE_WEIGHT: float = 0.1
    CORRECTION_WEIGHT: float = 1.0 - BASELINE_WEIGHT

    ceiling_properties: Properties
    window_left_properties: Properties
    window_right_properties: Properties

    executor: ThreadPoolExecutor

    def __init__(
        self,
        executor: ThreadPoolExecutor,
        ceiling_properties: Properties,
        window_left_properties: Properties,
        window_right_properties: Properties
    ) -> None:
        self.executor = executor
        self.ceiling_properties = ceiling_properties
        self.window_left_properties = window_left_properties
        self.window_right_properties = window_right_properties

    # TODO: implement
    def predict(
        self,
        ceiling_pose: pose.Result | None,
        window_left_pose: pose.Result | None,
        window_right_pose: pose.Result | None,
        window_left_face: face.Result | None,
        window_right_face: face.Result | None
    ):
        raise NotImplementedError()

    # TODO: JIT estimations
    def __predict(
        self,
        ceiling_pose: list[pose.Result | None],
        window_left_pose: list[pose.Result | None] | None,
        window_right_pose: list[pose.Result | None] | None,
        window_left_face: list[face.Result | None] | None,
        window_right_face: list[face.Result | None] | None
    ) -> list[Result | None] | None:
        collective_centres, baseline_vectors = ceiling_baseline.estimate(ceiling_pose, None)

        left_corrections = (
            side_correction.estimate(
                ceiling_pose,
                window_left_pose,
                window_left_face,
                Perspective.WINDOW_LEFT,
                self.ceiling_properties,
                self.window_left_properties
            )
            if window_left_pose is not None
            and window_left_face is not None
            else None
        )

        right_corrections = (
            side_correction.estimate(
                ceiling_pose,
                window_right_pose,
                window_right_face,
                Perspective.WINDOW_RIGHT,
                self.ceiling_properties,
                self.window_right_properties
            )
            if window_right_pose is not None
            and window_right_face is not None
            else None
        )

        result_vectors = baseline_vectors

        # print(f'\n{left_corrections = }')
        # print(f'\n{right_corrections = }\n')

        if left_corrections is not None:
            result_vectors += left_corrections

        if right_corrections is not None:
            result_vectors += right_corrections

        return [
            Result(centres, vectors)
            for (centres, vectors)
            in zip(
                collective_centres,
                result_vectors
            )
        ]

    # NOTE: heuristic idea: actors seen from right and left are in reversed lexicographic order
    @autostart
    async def stream(self) -> Fiber[Input | None, list[Result | None] | None]:
        executor = self.executor
        loop = asyncio.get_running_loop()

        results: list[Result | None] | None = None

        while True:
            match (yield results):
                case (
                    list(ceiling_pose),
                    window_left_pose,
                    window_right_pose,
                    window_left_face,
                    window_right_face
                ):
                    results = await loop.run_in_executor(
                        executor,
                        lambda: self.__predict(
                            ceiling_pose,
                            window_left_pose,
                            window_right_pose,
                            window_left_face,
                            window_right_face
                        )
                    )

                case _:
                    results = None
