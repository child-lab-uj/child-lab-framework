import asyncio
from concurrent.futures import ThreadPoolExecutor

from .core import hardware
from .core.flow import Machinery
from .core.video import Format, Perspective, Reader, Writer
from .logging import Logger
from .task import depth, face, gaze, pose
from .task.camera.transformation import heuristic
from .task.visualization import Visualizer
from .typing.flow import Component
from .typing.stream import Fiber

BATCH_SIZE = 1


async def __step(components: dict[str, Component]) -> Fiber[None, bool]:
    depth_fiber = components['depth'].stream()
    await depth_fiber.asend(None)
    ceiling_pose_fiber = components['ceiling_pose'].stream()
    await ceiling_pose_fiber.asend(None)
    window_left_pose_fiber = components['window_left_pose'].stream()
    await window_left_pose_fiber.asend(None)
    window_right_pose_fiber = components['window_right_pose'].stream()
    await window_right_pose_fiber.asend(None)
    window_left_to_ceiling_fiber = components['window_left_to_ceiling'].stream()
    await window_left_to_ceiling_fiber.asend(None)
    window_right_to_ceiling_fiber = components['window_right_to_ceiling'].stream()
    await window_right_to_ceiling_fiber.asend(None)
    window_left_face_fiber = components['window_left_face'].stream()
    await window_left_face_fiber.asend(None)
    window_right_face_fiber = components['window_right_face'].stream()
    await window_right_face_fiber.asend(None)
    window_left_gaze_fiber = components['window_left_gaze'].stream()
    await window_left_gaze_fiber.asend(None)
    window_right_gaze_fiber = components['window_right_gaze'].stream()
    await window_right_gaze_fiber.asend(None)
    ceiling_gaze_fiber = components['ceiling_gaze'].stream()
    await ceiling_gaze_fiber.asend(None)
    visualizer_fiber = components['visualizer'].stream()
    await visualizer_fiber.asend(None)
    writer_fiber = components['writer'].stream()
    await writer_fiber.asend(None)
    ceiling_reader_fiber = components['ceiling_reader'].stream()
    await ceiling_reader_fiber.asend(None)
    window_left_reader_fiber = components['window_left_reader'].stream()
    await window_left_reader_fiber.asend(None)
    window_right_reader_fiber = components['window_right_reader'].stream()
    await window_right_reader_fiber.asend(None)

    while (
        (ceiling_reader_value := await ceiling_reader_fiber.asend(None)) is not None
        and (window_left_reader_value := await window_left_reader_fiber.asend(None))
        is not None
        and (window_right_reader_value := await window_right_reader_fiber.asend(None))
        is not None
    ):
        (
            ceiling_pose_value,
            window_left_pose_value,
            window_right_pose_value,
        ) = await asyncio.gather(
            ceiling_pose_fiber.asend(ceiling_reader_value),
            window_left_pose_fiber.asend(window_left_reader_value),
            window_right_pose_fiber.asend(window_right_reader_value),
        )

        ceiling_depth_value = await depth_fiber.asend(ceiling_reader_value)
        window_left_depth_value = await depth_fiber.asend(window_left_reader_value)
        window_right_depth_value = await depth_fiber.asend(window_right_reader_value)

        (
            window_left_face_value,
            window_right_face_value,
            window_left_to_ceiling_value,
            window_right_to_ceiling_value,
        ) = await asyncio.gather(
            window_left_face_fiber.asend(
                (window_left_reader_value, window_left_pose_value)
            ),
            window_right_face_fiber.asend(
                (window_right_reader_value, window_right_pose_value)
            ),
            window_left_to_ceiling_fiber.asend(
                (
                    window_left_pose_value,
                    ceiling_pose_value,
                    window_left_depth_value,
                    ceiling_depth_value,
                )
            ),
            window_right_to_ceiling_fiber.asend(
                (
                    window_right_pose_value,
                    ceiling_pose_value,
                    window_right_depth_value,
                    ceiling_depth_value,
                )
            ),
        )

        (window_left_gaze_value, window_right_gaze_value) = await asyncio.gather(
            window_left_gaze_fiber.asend(
                (window_left_reader_value, window_left_face_value)
            ),
            window_right_gaze_fiber.asend(
                (window_right_reader_value, window_right_face_value)
            ),
        )

        ceiling_gaze_value = await ceiling_gaze_fiber.asend(
            (
                ceiling_pose_value,
                window_left_gaze_value,
                window_right_gaze_value,
                window_left_to_ceiling_value,
                window_right_to_ceiling_value,
            )
        )

        visualizer_value = await visualizer_fiber.asend(
            (ceiling_reader_value, ceiling_pose_value, ceiling_gaze_value)
        )

        await writer_fiber.asend((visualizer_value))

        yield False

    while True:
        yield True


async def main() -> None:
    executor = ThreadPoolExecutor(max_workers=1)
    device = hardware.get_best_device()

    ceiling_reader = Reader(
        'dev/data/aruco_cubic_ultra_short/ceiling.mp4',
        perspective=Perspective.CEILING,
        batch_size=BATCH_SIZE,
    )

    window_left_reader = Reader(
        'dev/data/aruco_cubic_ultra_short/window_left.mp4',
        perspective=Perspective.WINDOW_LEFT,
        batch_size=BATCH_SIZE,
        like=ceiling_reader.properties,
    )

    window_right_reader = Reader(
        'dev/data/aruco_cubic_ultra_short/window_right.mp4',
        perspective=Perspective.WINDOW_RIGHT,
        batch_size=BATCH_SIZE,
        like=ceiling_reader.properties,
    )

    depth_estimator = depth.Estimator(
        executor,
        device,
        input=ceiling_reader.properties,
    )

    ceiling_pose_estimator = pose.Estimator(
        executor,
        max_detections=2,
        threshold=0.5,
    )

    window_left_pose_estimator = pose.Estimator(
        executor,
        max_detections=2,
        threshold=0.5,
    )

    window_right_pose_estimator = pose.Estimator(
        executor,
        max_detections=2,
        threshold=0.5,
    )

    window_left_to_ceiling_transformation_estimator = heuristic.Estimator(
        executor,
        window_left_reader.properties,
        ceiling_reader.properties,
        keypoint_threshold=0.5,
    )

    window_right_to_ceiling_transformation_estimator = heuristic.Estimator(
        executor,
        window_right_reader.properties,
        ceiling_reader.properties,
        keypoint_threshold=0.5,
    )

    window_left_face_estimator = face.Estimator(executor, threshold=0.1)

    window_right_face_estimator = face.Estimator(executor, threshold=0.1)

    window_left_gaze_estimator = gaze.Estimator(
        executor,
        properties=window_left_reader.properties,
        wild=False,
        limit_angles=True,
    )

    window_right_gaze_estimator = gaze.Estimator(
        executor,
        properties=window_right_reader.properties,
        wild=True,
        limit_angles=True,
    )

    gaze_projector = gaze.ceiling_projection.Estimator(
        executor,
        ceiling_reader.properties,
        window_left_reader.properties,
        window_right_reader.properties,
    )

    visualizer = Visualizer(
        executor,
        properties=ceiling_reader.properties,
        confidence_threshold=0.5,
    )

    writer = Writer(
        'dev/output/gaze_projection_test.mp4',
        ceiling_reader.properties,
        output_format=Format.MP4,
    )

    machinery = Machinery(
        [
            ('ceiling_reader', ceiling_reader),
            ('window_left_reader', window_left_reader),
            ('window_right_reader', window_right_reader),
            ('depth', depth_estimator, 'ceiling_reader'),
            ('ceiling_pose', ceiling_pose_estimator, 'ceiling_reader'),
            ('window_left_pose', window_left_pose_estimator, 'window_left_reader'),
            ('window_right_pose', window_right_pose_estimator, 'window_right_reader'),
            (
                'window_left_to_ceiling',
                window_left_to_ceiling_transformation_estimator,
                (
                    'window_left_pose',
                    'ceiling_pose',
                    'window_left_depth',
                    'ceiling_depth',
                ),
            ),
            (
                'window_right_to_ceiling',
                window_right_to_ceiling_transformation_estimator,
                (
                    'window_right_pose',
                    'ceiling_pose',
                    'window_right_depth',
                    'ceiling_depth',
                ),
            ),
            (
                'window_left_face',
                window_left_face_estimator,
                ('window_left_reader', 'window_left_pose'),
            ),
            (
                'window_right_face',
                window_right_face_estimator,
                ('window_right_reader', 'window_right_pose'),
            ),
            (
                'window_left_gaze',
                window_left_gaze_estimator,
                ('window_left_reader', 'window_left_face'),
            ),
            (
                'window_right_gaze',
                window_right_gaze_estimator,
                ('window_right_reader', 'window_right_face'),
            ),
            (
                'ceiling_gaze',
                gaze_projector,
                (
                    'ceiling_pose',
                    'window_left_pose',
                    'window_right_pose',
                    'window_left_gaze',
                    'window_right_gaze',
                    'window_left_to_ceiling',
                    'window_right_to_ceiling',
                ),
            ),
            (
                'visualizer',
                visualizer,
                (
                    'ceiling_reader',
                    'ceiling_pose',
                    'ceiling_gaze',
                ),
            ),
            ('writer', writer, 'visualizer'),
        ]
    )

    status = __step(machinery.components)

    step = 1

    while not (await status.asend(None)):
        Logger.info('step:', step)
        step += 1
