import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .core.video import Reader, Writer, Perspective, Properties, Format
from .task import pose, face, gaze
from .task.visualization import Visualizer
from .core.flow import Machinery


# def main() -> None:
#     # Open streams:
#     ceiling_reader_thread = ceiling_reader.stream()
#     window_left_reader_thread = window_left_reader.stream()
#     window_right_reader_thread = window_right_reader.stream()

#     writer_thread = writer.stream()

#     ceiling_pose_thread = ceiling_pose_estimator.stream()
#     window_left_pose_thread = window_left_pose_estimator.stream()
#     window_right_pose_thread = window_right_pose_estimator.stream()
#     window_left_face_thread = window_left_face_estimator.stream()
#     window_right_face_thread = window_right_face_estimator.stream()
#     gaze_thread = gaze_estimator.stream()

#     visualizer_thread = visualizer.stream()


#     frames_count = 0
#     fps = ceiling_reader.properties.fps

#     while (
#         (ceiling_frames := ceiling_reader_thread.send(None)) and
#         (window_left_frames := window_left_reader_thread.send(None)) and
#         (window_right_frames := window_right_reader_thread.send(None))
#     ):
#         ceiling_poses = ceiling_pose_thread.send(ceiling_frames)
#         window_left_poses = window_left_pose_thread.send(window_left_frames)
#         window_right_poses = window_right_pose_thread.send(window_right_frames)

#         window_left_faces = window_left_face_thread.send((window_left_frames, window_left_poses))
#         window_right_faces = window_right_face_thread.send((window_right_frames, window_right_poses))

#         assert ceiling_poses is not None
#         assert window_left_poses is not None
#         assert window_right_poses is not None
#         assert window_left_faces is not None
#         assert window_right_faces is not None

#         gazes = gaze_thread.send((
#             ceiling_poses,
#             window_left_poses,
#             window_right_poses,
#             window_left_faces,
#             window_right_faces,
#         ))

#         annotated_frames = visualizer_thread.send((ceiling_frames, ceiling_poses, gazes))
#         writer_thread.send(annotated_frames)

#         frames_count += len(ceiling_frames)
#         seconds = frames_count / fps

#         sys.stdout.write(f'\r{seconds = :.2f}')
#         sys.stdout.flush()


async def main() -> None:
    executor = ThreadPoolExecutor(max_workers=8)

    ceiling_reader = Reader(
        'dev/data/ultra_short/ceiling.mp4',
        perspective=Perspective.CEILING,
        batch_size=5
    )

    ceiling_properties = ceiling_reader.properties

    window_left_reader = Reader(
        'dev/data/short/window_left.mp4',
        perspective=Perspective.WINDOW_LEFT,
        batch_size=5
    )

    window_right_reader = Reader(
        'dev/data/short/window_right.mp4',
        perspective=Perspective.WINDOW_RIGHT,
        batch_size=5
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
