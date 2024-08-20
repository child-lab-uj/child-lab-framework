import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .core.video import Reader, Writer, Perspective, Properties, Format
from .task import pose, face, gaze
from .task.visualization import Visualizer
from .core.flow import Machinery

BATCH_SIZE = 15


async def main() -> None:
    executor = ThreadPoolExecutor(max_workers=8)

    ceiling_reader = Reader(
        'dev/data/ultra_short/ceiling.mp4',
        perspective=Perspective.CEILING,
        batch_size=BATCH_SIZE
    )

    ceiling_properties = ceiling_reader.properties

    window_left_reader = Reader(
        'dev/data/short/window_left.mp4',
        perspective=Perspective.WINDOW_LEFT,
        batch_size=BATCH_SIZE
    )

    window_right_reader = Reader(
        'dev/data/short/window_right.mp4',
        perspective=Perspective.WINDOW_RIGHT,
        batch_size=BATCH_SIZE
    )

    ceiling_pose_estimator = pose.Estimator(
        executor,
        max_detections=2,
        threshold=0.5
    )

    window_left_pose_estimator = pose.Estimator(
        executor,
        max_detections=2,
        threshold=0.5
    )

    window_right_pose_estimator = pose.Estimator(
        executor,
        max_detections=2,
        threshold=0.5
    )

    window_left_face_estimator = face.Estimator(
        executor,
        max_results=2,
        detection_threshold=0.1,
        tracking_threshold=0.1
    )

    window_right_face_estimator = face.Estimator(
        executor,
        max_results=2,
        detection_threshold=0.1,
        tracking_threshold=0.1
    )

    gaze_estimator = gaze.Estimator(
        executor,
        ceiling_reader.properties,
        window_left_reader.properties,
        window_right_reader.properties,
    )

    visualizer = Visualizer(executor, confidence_threshold=0.5)

    writer = Writer(
        'dev/output/gaze_test.mp4',
        ceiling_reader.properties,
        output_format=Format.MP4
    )

    machinery=Machinery([
        ('ceiling_reader', ceiling_reader),
        ('window_left_reader', window_left_reader),
        ('window_right_reader', window_right_reader),
        ('ceiling_pose', ceiling_pose_estimator, 'ceiling_reader'),
        ('window_left_pose', window_left_pose_estimator, 'window_left_reader'),
        ('window_right_pose', window_right_pose_estimator, 'window_right_reader'),
        ('window_left_face', window_left_face_estimator, 'window_left_reader'),
        ('window_right_face', window_right_face_estimator, 'window_right_reader'),
        ('gaze', gaze_estimator, (
            'ceiling_pose','window_left_pose', 'window_right_pose',
            'window_left_face', 'window_right_face'
        )),
        ('visualizer', visualizer, ('ceiling_reader', 'ceiling_pose', 'gaze')),
        ('writer', writer, 'visualizer')
    ])

    await machinery.run()
