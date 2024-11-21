import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from ..core.video import Format, Input, Reader, Writer
from ..logging import Logger
from ..task import depth, face, gaze, pose
from ..task.camera import transformation
from ..task.visualization import Configuration as VisualizationConfiguration
from ..task.visualization import Visualizer

BATCH_SIZE = 32


def main(
    inputs: tuple[Input, Input, Input], device: torch.device, output_directory: Path
) -> None:
    # ignore exceeded allocation limit on MPS and CUDA - very important!
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    executor = ThreadPoolExecutor(max_workers=8)

    ceiling, window_left, window_right = inputs

    ceiling_reader = Reader(
        ceiling,
        batch_size=BATCH_SIZE,
    )
    ceiling_properties = ceiling_reader.properties

    window_left_reader = Reader(
        window_left,
        batch_size=BATCH_SIZE,
        height=ceiling_properties.height,
        width=ceiling_properties.width,
        fps=ceiling_properties.fps,
    )

    window_right_reader = Reader(
        window_right,
        batch_size=BATCH_SIZE,
        height=ceiling_properties.height,
        width=ceiling_properties.width,
        fps=ceiling_properties.fps,
    )

    depth_estimator = depth.Estimator(executor, device, input=ceiling_reader.properties)

    transformation_estimator = transformation.heuristic.Estimator(
        executor,
        window_left_reader.properties,
        ceiling_reader.properties,
        keypoint_threshold=0.35,
    )

    pose_estimator = pose.Estimator(
        executor,
        device,
        input=ceiling_reader.properties,
        max_detections=2,
        threshold=0.5,
    )

    face_estimator = face.Estimator(
        executor,
        # A workaround to use the model efficiently on both desktop and server.
        # TODO: remove this as soon as it's possible to specify device per component via CLI/config file.
        device if device == torch.device('cuda') else torch.device('cpu'),
        input=ceiling_reader.properties,
        confidence_threshold=0.5,
        suppression_threshold=0.1,
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

    ceiling_visualizer = Visualizer(
        executor,
        properties=ceiling_reader.properties,
        configuration=VisualizationConfiguration(),
    )

    window_left_visualizer = Visualizer(
        executor,
        properties=window_left_reader.properties,
        configuration=VisualizationConfiguration(),
    )

    window_right_visualizer = Visualizer(
        executor,
        properties=window_right_reader.properties,
        configuration=VisualizationConfiguration(),
    )

    ceiling_writer = Writer(
        output_directory / (ceiling.name + '.mp4'),
        ceiling_reader.properties,
        output_format=Format.MP4,
    )

    window_left_writer = Writer(
        output_directory / (window_left.name + '.mp4'),
        window_left_reader.properties,
        output_format=Format.MP4,
    )

    window_right_writer = Writer(
        output_directory / (window_right.name + '.mp4'),
        window_right_reader.properties,
        output_format=Format.MP4,
    )

    Logger.info('Components instantiated')

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

        Logger.info('Estimating poses...')
        ceiling_poses = pose_estimator.predict_batch(ceiling_frames)
        window_left_poses = pose_estimator.predict_batch(window_left_frames)
        window_right_poses = pose_estimator.predict_batch(window_right_frames)
        Logger.info('Done!')

        if ceiling_poses is None:
            Logger.error('ceiling_poses == None')

        if window_left_poses is None:
            Logger.error('window_left_poses == None')

        if window_right_poses is None:
            Logger.error('window_right_poses == None')

        Logger.info('Estimating depth...')
        ceiling_depth = depth_estimator.predict(ceiling_frames[0])
        ceiling_depths = [ceiling_depth for _ in range(n_frames)]
        Logger.info('Done!')

        Logger.info('Estimating transformations...')
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
        Logger.info('Done!')

        Logger.info('Detecting faces...')
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
        Logger.info('Done!')

        if window_left_faces is None:
            Logger.error('window_left_faces == None')

        if window_right_faces is None:
            Logger.error('window_right_faces == None')

        Logger.info('Estimating gazes...')
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
        Logger.info('Done!')

        Logger.info('Visualizing results...')
        ceiling_annotated_frames = ceiling_visualizer.annotate_batch(
            ceiling_frames,
            ceiling_poses,
            ceiling_gazes,
        )

        window_left_annotated_frames = window_left_visualizer.annotate_batch(
            window_left_frames,
            window_left_poses,
            window_left_faces,
            window_left_gazes,
        )

        window_right_annotated_frames = window_right_visualizer.annotate_batch(
            window_right_frames,
            window_right_poses,
            window_right_faces,
            window_right_gazes,
        )
        Logger.info('Done!')

        Logger.info('Saving results...')
        ceiling_writer.write_batch(ceiling_annotated_frames)
        window_left_writer.write_batch(window_left_annotated_frames)
        window_right_writer.write_batch(window_right_annotated_frames)
        Logger.info('Done!')

        Logger.info('Step complete')
