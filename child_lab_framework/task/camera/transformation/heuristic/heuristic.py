import asyncio
from concurrent.futures.thread import ThreadPoolExecutor
from itertools import starmap

import cv2

from .....core.sequence import imputed_with_reference_inplace
from .....core.transformation import ProjectiveTransformation
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
        self.to_view_intrinsics = to_view.calibration.intrinsics[:3, :3]
        self.to_view_distortion = to_view.calibration.distortion

        self.keypoint_threshold = keypoint_threshold

        self.executor = executor

    def predict(
        self,
        from_pose: pose.Result,
        to_pose: pose.Result,
        from_depth: FloatArray2,
        to_depth: FloatArray2,
    ) -> ProjectiveTransformation | None:
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
                return ProjectiveTransformation(
                    cv2.Rodrigues(rotation)[0],  # type: ignore
                    translation,
                    self.to_view.calibration,
                )

            case None:
                return None

    def predict_batch(
        self,
        from_poses: list[pose.Result],
        to_poses: list[pose.Result],
        from_depths: list[FloatArray2],
        to_depths: list[FloatArray2],
    ) -> list[ProjectiveTransformation] | None:
        return imputed_with_reference_inplace(
            list(
                starmap(
                    self.predict,
                    zip(
                        from_poses,
                        to_poses,
                        from_depths,
                        to_depths,
                    ),
                )
            )
        )

    def __predict_safe(
        self,
        from_pose: pose.Result | None,
        to_pose: pose.Result | None,
        from_depth: FloatArray2,
        to_depth: FloatArray2,
    ) -> ProjectiveTransformation | None:
        if from_pose is None or to_pose is None:
            return None

        return self.predict(from_pose, to_pose, from_depth, to_depth)

    async def stream(
        self,
    ) -> Fiber[Input | None, list[ProjectiveTransformation | None] | None]:
        loop = asyncio.get_running_loop()
        executor = self.executor

        results: list[ProjectiveTransformation | None] | None = None

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
