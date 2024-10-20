import os
from concurrent.futures import ThreadPoolExecutor

import torch

from .core.video import Format, Perspective, Reader, Writer
from .logging import Logger
from .task import depth, face, gaze, pose
from .task.camera import transformation
from .task.visualization import Visualizer

BATCH_SIZE = 32


def main() -> None:
    # ignore exceeded allocation limit on MPS - very important!
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

    executor = ThreadPoolExecutor(max_workers=8)
    gpu = torch.device('mps')

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

    depth_estimator = depth.Estimator(executor, gpu, input=ceiling_reader.properties)

    transformation_estimator = transformation.heuristic.Estimator(
        executor,
        window_left_reader.properties,
        ceiling_reader.properties,
        keypoint_threshold=0.35,
    )

    pose_estimator = pose.Estimator(
        executor,
        gpu,
        input=ceiling_reader.properties,
        max_detections=2,
        threshold=0.5,
    )

    face_estimator = face.Estimator(
        executor,
        input=ceiling_reader.properties,
        threshold=0.1,
    )

    window_left_gaze_estimator = gaze.Estimator(
        executor,
        input=window_left_reader.properties,
    )

    window_right_gaze_estimator = gaze.Estimator(
        executor,
        input=window_right_reader.properties,
    )

    ceiling_gaze_estimator = gaze.ceiling_projection.Estimator(
        executor,
        ceiling_reader.properties,
        window_left_reader.properties,
        window_right_reader.properties,
    )

    # social_distance_estimator = social_distance.Estimator(executor)
    # social_distance_logger = social_distance.FileLogger('dev/output/distance.csv')

    visualizer = Visualizer(
        executor,
        properties=window_left_reader.properties,
        confidence_threshold=0.5,
    )

    ceiling_writer = Writer(
        'dev/output/sequential/ceiling.mp4',
        ceiling_reader.properties,
        output_format=Format.MP4,
    )

    window_left_writer = Writer(
        'dev/output/sequential/window_left.mp4',
        window_left_reader.properties,
        output_format=Format.MP4,
    )

    window_right_writer = Writer(
        'dev/output/sequential/window_right.mp4',
        window_right_reader.properties,
        output_format=Format.MP4,
    )

    while True:
        ceiling_frames = ceiling_reader.read_batch()
        if ceiling_frames is None:
            break

        window_left_frames = window_left_reader.read_batch()
        if window_left_frames is None:
            break

        window_right_frames = window_right_reader.read_batch()
        if window_right_frames is None:
            break

        n_frames = len(ceiling_frames)

        ceiling_poses = pose_estimator.predict_batch(ceiling_frames)
        window_left_poses = pose_estimator.predict_batch(window_left_frames)
        window_right_poses = pose_estimator.predict_batch(window_right_frames)

        ceiling_depth = depth_estimator.predict(ceiling_frames[0])
        ceiling_depths = [ceiling_depth for _ in range(n_frames)]

        window_left_to_ceiling = (
            transformation_estimator.predict_batch(
                ceiling_poses,
                window_left_poses,
                ceiling_depths,
                [None for _ in range(n_frames)],  # type: ignore  # safe to pass
            )
            if ceiling_poses is not None and window_left_poses is not None
            else None
        )

        window_right_to_ceiling = (
            transformation_estimator.predict_batch(
                ceiling_poses,
                window_right_poses,
                ceiling_depths,
                [None for _ in range(n_frames)],  # type: ignore  # safe to pass
            )
            if ceiling_poses is not None and window_right_poses is not None
            else None
        )

        if ceiling_poses is None:
            Logger.error('ceiling_poses == None')

        if window_left_poses is None:
            Logger.error('window_left_poses == None')

        if window_right_poses is None:
            Logger.error('window_right_poses == None')

        window_left_faces = (
            face_estimator.predict_batch(window_left_frames, window_left_poses)
            if window_left_poses is not None
            else None
        )

        window_right_faces = (
            face_estimator.predict_batch(window_right_frames, window_right_poses)
            if window_right_poses is not None
            else None
        )

        if window_left_faces is None:
            Logger.error('window_left_faces == None')

        if window_right_faces is None:
            Logger.error('window_right_faces == None')

        window_left_gazes = (
            window_left_gaze_estimator.predict_batch(
                window_left_frames, window_left_faces
            )
            if window_left_faces is not None
            else None
        )

        window_right_gazes = (
            window_right_gaze_estimator.predict_batch(
                window_right_frames, window_right_faces
            )
            if window_right_faces is not None
            else None
        )

        ceiling_gazes = (
            ceiling_gaze_estimator.predict_batch(
                ceiling_poses,
                window_left_gazes,
                window_right_gazes,
                window_left_to_ceiling,
                window_right_to_ceiling,
            )
            if ceiling_poses is not None
            else None
        )

        ceiling_annotated_frames = visualizer.annotate_batch(
            ceiling_frames,
            ceiling_poses,
            None,
            ceiling_gazes,
        )

        window_left_annotated_frames = visualizer.annotate_batch(
            window_left_frames,
            window_left_poses,
            window_left_faces,
            None,
        )

        window_right_annotated_frames = visualizer.annotate_batch(
            window_right_frames,
            window_right_poses,
            window_right_faces,
            None,
        )

        ceiling_writer.write_batch(ceiling_annotated_frames)
        window_left_writer.write_batch(window_left_annotated_frames)
        window_right_writer.write_batch(window_right_annotated_frames)

        Logger.info('Step complete')
