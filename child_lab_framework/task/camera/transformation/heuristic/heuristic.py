import asyncio
from concurrent.futures.thread import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import starmap

import numpy as np

from .....core.video import Properties
from .....typing.array import FloatArray1, FloatArray2
from .....typing.stream import Fiber
from .... import pose
from . import projection

type Input = tuple[
    list[pose.Result | None] | None,
    list[pose.Result | None] | None,
    list[FloatArray2],
    list[FloatArray2],
]


@dataclass(repr=False, frozen=True)
class Result:
    rotation: FloatArray2
    translation: FloatArray1

    def __repr__(self) -> str:
        rotation = self.rotation
        translation = self.translation
        return f'Result:\n{translation = }\n{rotation = }'


class Estimator:
    from_view: Properties

    to_view: Properties
    to_view_intrinsics: FloatArray2
    to_view_distortion: FloatArray1

    keypoint_threshold: float

    executor: ThreadPoolExecutor

    def __init__(
        self,
        executor: ThreadPoolExecutor,
        from_view: Properties,
        to_view: Properties,
        *,
        keypoint_threshold: float = 0.25,
    ) -> None:
        self.from_view = from_view

        self.to_view = to_view
        self.to_view_intrinsics = to_view.calibration.intrinsics()[:3, :3]
        self.to_view_distortion = to_view.calibration.distortion()

        self.keypoint_threshold = keypoint_threshold

        self.executor = executor

    def __predict_safe(
        self,
        from_pose: pose.Result | None,
        to_pose: pose.Result | None,
        from_depth: FloatArray2,
        to_depth: FloatArray2,
    ) -> Result | None:
        if from_pose is None or to_pose is None:
            return None

        match projection.estimate(
            from_pose,
            to_pose,
            from_depth,
            to_depth,
            self.to_view_intrinsics,
            self.to_view_distortion,
            self.keypoint_threshold,
        ):
            case rotation, translation:
                return Result(np.linalg.inv(rotation), -translation)

            case None:
                return None

    async def stream(self) -> Fiber[Input | None, list[Result | None] | None]:
        loop = asyncio.get_running_loop()
        executor = self.executor

        results: list[Result | None] | None = None

        while True:
            match (yield results):
                case list(from_poses), list(to_poses), list(from_depths), list(to_depths):
                    results = await loop.run_in_executor(
                        executor,
                        lambda: list(
                            starmap(
                                self.__predict_safe,
                                zip(from_poses, to_poses, from_depths, to_depths),
                            )
                        ),
                    )

                case _:
                    results = None
