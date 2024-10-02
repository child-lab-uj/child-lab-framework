import asyncio
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ...core.algebra import normalized_3d
from ...core.stream import autostart
from ...core.video import Properties
from ...typing.array import FloatArray1, FloatArray2, FloatArray3
from ...typing.stream import Fiber
from .. import face, pose
from ..camera.transformation import heuristic
from . import ceiling_baseline, side_correction

# Multi-camera gaze direction estimation without strict algebraic camera models:
# 1. estimate actor's skeleton on each frame in both cameras
#    (heuristic: adults' keypoints have higher variance, children have smaller bounding boxes)
# 2. compute a ceiling baseline vector (perpendicular to shoulder line in a ceiling camera)
# 3. detect actor's face on the other camera
# 4. compute the offset-baseline vector (normal to face, Wim's MediaPipe solution')
# 5. Rotate it to the ceiling camera's space and combine with the ceiling baseline


type Input = tuple[
    list[pose.Result | None] | None,
    list[pose.Result | None] | None,
    list[pose.Result | None] | None,
    list[face.Result | None] | None,
    list[face.Result | None] | None,
    list[heuristic.Result | None] | None,
    list[heuristic.Result | None] | None,
]


@dataclass
class Result:
    centres: FloatArray2
    versors: FloatArray2

    def iter(self) -> Iterable[tuple[FloatArray1, FloatArray1]]:
        return zip(self.centres, self.versors)


class Estimator:
    BASELINE_WEIGHT: float = 0.5
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
        window_right_properties: Properties,
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
        window_right_face: face.Result | None,
    ):
        raise NotImplementedError()

    # TODO: JIT estimations
    def __predict(
        self,
        ceiling_pose: list[pose.Result | None],
        window_left_pose: list[pose.Result | None] | None,
        window_right_pose: list[pose.Result | None] | None,
        window_left_face: list[face.Result | None] | None,
        window_right_face: list[face.Result | None] | None,
        window_left_to_ceiling: list[heuristic.Result | None] | None,
        window_right_to_ceiling: list[heuristic.Result | None] | None,
    ) -> list[Result | None] | None:
        baseline = ceiling_baseline.estimate(ceiling_pose, None)

        if baseline is None:
            return None

        collective_centres, baseline_vectors = baseline

        left_corrections = (
            side_correction.estimate(
                ceiling_pose, window_left_pose, window_left_face, window_left_to_ceiling
            )
            if window_left_pose is not None
            and window_left_face is not None
            and window_left_to_ceiling is not None
            else None
        )

        right_corrections = (
            side_correction.estimate(
                ceiling_pose,
                window_right_pose,
                window_right_face,
                window_right_to_ceiling,
            )
            if window_right_pose is not None
            and window_right_face is not None
            and window_right_to_ceiling is not None
            else None
        )

        baseline_vectors *= self.BASELINE_WEIGHT
        result_vectors: FloatArray3 = baseline_vectors

        match (left_corrections, right_corrections):
            case None, None:
                ...

            case left, None:
                result_vectors += left * self.CORRECTION_WEIGHT

            case None, right:
                result_vectors += right * self.CORRECTION_WEIGHT

            case left, right:
                weight = self.CORRECTION_WEIGHT / 2.0
                result_vectors += left * weight
                result_vectors += right * weight

        result_vectors = normalized_3d(result_vectors)

        return [
            Result(centres, vectors)
            for (centres, vectors) in zip(collective_centres, result_vectors)
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
                    window_right_face,
                    window_left_to_ceiling,
                    window_right_to_ceiling,
                ):
                    results = await loop.run_in_executor(
                        executor,
                        lambda: self.__predict(
                            ceiling_pose,
                            window_left_pose,
                            window_right_pose,
                            window_left_face,
                            window_right_face,
                            window_left_to_ceiling,
                            window_right_to_ceiling,
                        ),
                    )

                case _:
                    results = None
