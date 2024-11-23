import asyncio
from concurrent.futures.thread import ThreadPoolExecutor
from itertools import starmap

from .....core.sequence import imputed_with_reference_inplace
from .....core.transformation import Buffer, Transformation
from .....core.video import Properties
from .....logging import Logger
from .....typing.array import FloatArray2
from .....typing.stream import Fiber
from .... import pose
from . import box_kabsch, kabsch, projection

type Input = tuple[
    list[pose.Result | None] | None,
    list[pose.Result | None] | None,
    list[FloatArray2],
    list[FloatArray2],
]


class Estimator:
    executor: ThreadPoolExecutor
    transformation_buffer: Buffer[str]

    from_view: Properties
    to_view: Properties

    keypoint_threshold: float

    def __init__(
        self,
        executor: ThreadPoolExecutor,
        transformation_buffer: Buffer[str],
        from_view: Properties,
        to_view: Properties,
        *,
        keypoint_threshold: float = 0.25,
    ) -> None:
        self.executor = executor

        self.transformation_buffer = transformation_buffer
        transformation_buffer.add_frame_of_reference(from_view.name)
        transformation_buffer.add_frame_of_reference(to_view.name)

        self.from_view = from_view
        self.to_view = to_view

        self.keypoint_threshold = keypoint_threshold

    # TODO: rebuild transformation estimation API - get rid of `heuristic.Estimator` facade and implement appropriate estimators for concrete methods.
    def predict(
        self,
        from_pose: pose.Result,
        to_pose: pose.Result,
        from_depth: FloatArray2,
        to_depth: FloatArray2,
    ) -> Transformation | None:
        # TODO: remove this workaround when actors are recognized and matched between cameras
        if len(from_pose.actors) != len(to_pose.actors):
            return None

        from_calibration = self.from_view.calibration
        to_calibration = self.to_view.calibration

        from_pose_3d = from_pose.unproject(from_calibration, from_depth)
        to_pose_3d = to_pose.unproject(to_calibration, to_depth)

        from_name = self.from_view.name
        to_name = self.to_view.name

        buffer = self.transformation_buffer

        transformation: Transformation | None = None
        for transformation in [
            projection.estimate(
                from_pose,
                from_pose_3d,
                to_pose,
                to_pose_3d,
                from_calibration,
                to_calibration,
                self.keypoint_threshold,
            ),
            kabsch.estimate(
                from_pose,
                from_pose_3d,
                to_pose,
                to_pose_3d,
                self.keypoint_threshold,
            ),
            box_kabsch.estimate(
                from_pose,
                to_pose,
                from_depth,
                to_depth,
                from_calibration,
                to_calibration,
                self.keypoint_threshold,
            ),
        ]:
            if transformation is None:
                continue

            buffer.update_transformation_if_better(
                from_name,
                to_name,
                from_pose_3d,
                to_pose_3d,
                transformation,
            )

        error = buffer.reprojection_error(from_name, to_name, from_pose_3d, to_pose_3d)

        Logger.info(f'Reprojection error from {from_name} to {to_name}: {error:.2e}')

        return buffer[from_name, to_name]

    def predict_batch(
        self,
        from_poses: list[pose.Result],
        to_poses: list[pose.Result],
        from_depths: list[FloatArray2],
        to_depths: list[FloatArray2],
    ) -> list[Transformation] | None:
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
    ) -> Transformation | None:
        if from_pose is None or to_pose is None:
            return None

        return self.predict(from_pose, to_pose, from_depth, to_depth)

    async def stream(
        self,
    ) -> Fiber[Input | None, list[Transformation | None] | None]:
        loop = asyncio.get_running_loop()
        executor = self.executor

        results: list[Transformation | None] | None = None

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
