from concurrent.futures import ThreadPoolExecutor

from child_lab_framework.task.camera.transformation.heuristic import heuristic

from .core import hardware
from .core.video import Perspective, Reader
from .task import depth as dp
from .task import pose as ps

BATCH_SIZE = 1


async def main() -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    device = hardware.get_best_device()

    ceiling_reader = Reader(
        'dev/data/aruco_cubic_ultra_short/ceiling.mp4',
        perspective=Perspective.CEILING,
        batch_size=1,
    )
    ceiling = ceiling_reader.stream()
    await ceiling.asend(None)

    window_left_reader = Reader(
        'dev/data/aruco_cubic_ultra_short/window_left.mp4',
        perspective=Perspective.WINDOW_LEFT,
        batch_size=1,
        like=ceiling_reader.properties,
    )
    window_left = window_left_reader.stream()
    await window_left.asend(None)

    depth = dp.Estimator(executor, device, input=ceiling_reader.properties)
    pose = ps.Estimator(executor, max_detections=2, threshold=0.5)
    transformation_estimator = heuristic.Estimator(
        executor,
        from_view=ceiling_reader.properties,
        to_view=window_left_reader.properties,
        keypoint_threshold=0.85,
    )

    while (ceiling_frames := await ceiling.asend(None)) and (
        window_left_frames := await window_left.asend(None)
    ):
        ceiling_frame = ceiling_frames[0]
        window_left_frame = window_left_frames[0]

        ceiling_poses = pose.predict(ceiling_frame)
        assert ceiling_poses is not None

        window_left_poses = pose.predict(window_left_frame)
        assert window_left_poses is not None

        ceiling_depth = depth.predict(ceiling_frame)
        window_left_depth = depth.predict(window_left_frame)

        transformation_estimator.predict(
            ceiling_poses,
            window_left_poses,
            ceiling_depth,
            window_left_depth,
        )

        # assert transformation is not None

        # ceiling_points = ceiling_poses.depersonificated_keypoints.view()[:, :3]
        # ceiling_points[:, 2] = 1.0

        # window_left_points = window_left_poses.depersonificated_keypoints.view()[:, :3]
        # window_left_points[:, 2] = 1.0

        # rotation = transformation.rotation.T
        # translation = transformation.translation.reshape(1, -1)

        # ceiling_projected_points = window_left_points @ rotation + translation

        # diff = ceiling_points - ceiling_projected_points

        # print(f'{diff =}')
