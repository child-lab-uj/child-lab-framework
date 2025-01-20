import os
import typing
from itertools import repeat
from pathlib import Path

import torch

from child_lab_framework.postprocessing.imputation import (
    imputed_with_closest_known_reference,
)

from ..core import transformation
from ..core.file import save
from ..core.video import Format, Input, Reader, Writer
from ..logging import Logger
from ..task import depth, face, gaze, pose
from ..task.camera.transformation import heuristic as heuristic_transformation
from ..task.visualization import Configuration as VisualizationConfiguration
from ..task.visualization import Visualizer

BATCH_SIZE = 32


def main(
    inputs: tuple[Input, Input, Input],
    device: torch.device,
    output_directory: Path,
    skip: int | None,
    transformation_buffer: transformation.Buffer[str] | None,
    dynamic_transformations: bool,
) -> None:
    # ignore exceeded allocation limit on MPS and CUDA - very important!
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
    window_left_properties = window_left_reader.properties

    window_right_reader = Reader(
        window_right,
        batch_size=BATCH_SIZE,
        height=ceiling_properties.height,
        width=ceiling_properties.width,
        fps=ceiling_properties.fps,
    )
    window_right_properties = window_right_reader.properties

    depth_estimator = depth.Estimator(device)

    transformation_buffer = transformation_buffer or transformation.Buffer()

    window_left_to_ceiling_transformation_estimator = heuristic_transformation.Estimator(
        transformation_buffer,
        window_left_properties,
        ceiling_properties,
        keypoint_threshold=0.35,
    )

    window_right_to_ceiling_transformation_estimator = heuristic_transformation.Estimator(
        transformation_buffer,
        window_right_properties,
        ceiling_properties,
        keypoint_threshold=0.35,
    )

    pose_estimator = pose.Estimator(
        device,
        input=ceiling_properties,
        max_detections=2,
        threshold=0.5,
    )

    face_estimator = face.Estimator(
        # A workaround to use the model efficiently on both desktop and server.
        # TODO: remove this as soon as it's possible to specify device per component via CLI/config file.
        device if device == torch.device('cuda') else torch.device('cpu'),
        input=ceiling_properties,
        confidence_threshold=0.5,
        suppression_threshold=0.1,
    )

    window_left_gaze_estimator = gaze.Estimator(
        input=window_left_properties,
    )

    window_right_gaze_estimator = gaze.Estimator(
        input=window_right_properties,
    )

    ceiling_gaze_estimator = gaze.ceiling_projection.Estimator(
        transformation_buffer,
        ceiling_properties,
        window_left_properties,
        window_right_properties,
    )

    # social_distance_estimator = social_distance.Estimator(executor)
    # social_distance_logger = social_distance.FileLogger('dev/output/distance.csv')

    ceiling_visualizer = Visualizer(
        properties=ceiling_properties,
        configuration=VisualizationConfiguration(),
    )

    window_left_visualizer = Visualizer(
        properties=window_left_properties,
        configuration=VisualizationConfiguration(),
    )

    window_right_visualizer = Visualizer(
        properties=window_right_properties,
        configuration=VisualizationConfiguration(),
    )

    ceiling_writer = Writer(
        output_directory / (ceiling.name + '.mp4'),
        ceiling_properties,
        output_format=Format.MP4,
    )

    ceiling_projection_writer = Writer(
        output_directory / (ceiling.name + '_projections.mp4'),
        ceiling_properties,
        output_format=Format.MP4,
    )

    ceiling_depth_writer = Writer(
        output_directory / (ceiling.name + '_depth.mp4'),
        ceiling_properties,
        output_format=Format.MP4,
    )

    window_left_depth_writer = Writer(
        output_directory / (window_left.name + '_depth.mp4'),
        window_left_properties,
        output_format=Format.MP4,
    )

    window_right_depth_writer = Writer(
        output_directory / (window_right.name + '_depth.mp4'),
        window_right_properties,
        output_format=Format.MP4,
    )

    window_left_writer = Writer(
        output_directory / (window_left.name + '.mp4'),
        window_left_properties,
        output_format=Format.MP4,
    )

    window_right_writer = Writer(
        output_directory / (window_right.name + '.mp4'),
        window_right_properties,
        output_format=Format.MP4,
    )

    Logger.info('Components instantiated')

    if skip is not None and skip > 0:
        frames_to_skip = skip * ceiling_properties.fps
        ceiling_reader.read_skipping(frames_to_skip)
        window_left_reader.read_skipping(frames_to_skip)
        window_right_reader.read_skipping(frames_to_skip)

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
        ceiling_poses = imputed_with_closest_known_reference(
            pose_estimator.predict_batch(ceiling_frames)
        )
        window_left_poses = imputed_with_closest_known_reference(
            pose_estimator.predict_batch(window_left_frames)
        )
        window_right_poses = imputed_with_closest_known_reference(
            pose_estimator.predict_batch(window_right_frames)
        )
        Logger.info('Done!')

        if ceiling_poses is None:
            Logger.error('ceiling_poses == None')

        if window_left_poses is None:
            Logger.error('window_left_poses == None')

        if window_right_poses is None:
            Logger.error('window_right_poses == None')

        Logger.info('Estimating depth...')
        ceiling_depth = depth_estimator.predict(
            ceiling_frames[0],
            ceiling_properties,
        )
        window_left_depth = depth_estimator.predict(
            window_left_frames[0],
            window_left_properties,
        )
        window_right_depth = depth_estimator.predict(
            window_right_frames[0],
            window_right_properties,
        )

        ceiling_depths = [ceiling_depth for _ in range(n_frames)]
        window_left_depths = [window_left_depth for _ in range(n_frames)]
        window_right_depths = [window_right_depth for _ in range(n_frames)]
        Logger.info('Done!')

        window_left_to_ceiling = None
        window_right_to_ceiling = None

        if dynamic_transformations:
            Logger.info('Estimating transformations...')
            window_left_to_ceiling = (
                imputed_with_closest_known_reference(
                    window_left_to_ceiling_transformation_estimator.predict_batch(
                        ceiling_poses,
                        window_left_poses,
                        ceiling_depths,
                        window_left_depths,
                    )
                )
                if ceiling_poses is not None and window_left_poses is not None
                else None
            )

            window_right_to_ceiling = (
                imputed_with_closest_known_reference(
                    window_right_to_ceiling_transformation_estimator.predict_batch(
                        ceiling_poses,
                        window_right_poses,
                        ceiling_depths,
                        window_right_depths,
                    )
                )
                if ceiling_poses is not None and window_right_poses is not None
                else None
            )
            Logger.info('Done!')

            if window_left_to_ceiling is None:
                Logger.error('window_left_to_ceiling == None')

            if window_right_to_ceiling is None:
                Logger.error('window_right_to_ceiling == None')

        Logger.info('Detecting faces...')
        window_left_faces = (
            imputed_with_closest_known_reference(
                face_estimator.predict_batch(window_left_frames, window_left_poses)
            )
            if window_left_poses is not None
            else None
        )

        window_right_faces = (
            imputed_with_closest_known_reference(
                face_estimator.predict_batch(window_right_frames, window_right_poses)
            )
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
            imputed_with_closest_known_reference(
                window_left_gaze_estimator.predict_batch(
                    window_left_frames, window_left_faces
                )
            )
            if window_left_faces is not None
            else None
        )

        window_right_gazes = (
            imputed_with_closest_known_reference(
                window_right_gaze_estimator.predict_batch(
                    window_right_frames, window_right_faces
                )
            )
            if window_right_faces is not None
            else None
        )

        ceiling_gazes = (
            imputed_with_closest_known_reference(
                ceiling_gaze_estimator.predict_batch(
                    ceiling_poses,
                    window_left_gazes,
                    window_right_gazes,
                    None,
                    None,
                )
            )
            if ceiling_poses is not None
            else None
        )
        Logger.info('Done!')

        if window_left_gazes is None:
            Logger.error('window_left_gazes == None')

        if window_right_gazes is None:
            Logger.error('window_right_gazes == None')

        Logger.info('Visualizing results...')

        if not dynamic_transformations:
            # Those variables are going to be used as if they were list;
            # tricking the type-checker saves a lot of boilerplate

            window_left_to_ceiling = typing.cast(
                list[transformation.Transformation],
                repeat(transformation_buffer['window_left', 'ceiling']),
            )
            window_right_to_ceiling = typing.cast(
                list[transformation.Transformation],
                repeat(transformation_buffer['window_right', 'ceiling']),
            )

        ceiling_projection_annotated_frames = ceiling_visualizer.annotate_batch(
            ceiling_frames,
            [
                p.unproject(window_left_properties.calibration, ceiling_depth)
                .transform(t.inverse)
                .project(ceiling_properties.calibration)
                if t is not None
                else None
                for p, t in zip(
                    window_left_poses or [],
                    window_left_to_ceiling or [],
                )
            ],
            [
                p.unproject(window_right_properties.calibration, ceiling_depth)
                .transform(t.inverse)
                .project(ceiling_properties.calibration)
                if t is not None
                else None
                for p, t in zip(
                    window_right_poses or [],
                    window_right_to_ceiling or [],
                )
            ],
        )

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
        ceiling_projection_writer.write_batch(ceiling_projection_annotated_frames)

        ceiling_depth_writer.write_batch([depth.to_frame(d) for d in ceiling_depths])
        window_left_depth_writer.write_batch(
            [depth.to_frame(d) for d in window_left_depths]
        )
        window_right_depth_writer.write_batch(
            [depth.to_frame(d) for d in window_right_depths]
        )

        ceiling_writer.write_batch(ceiling_annotated_frames)
        window_left_writer.write_batch(window_left_annotated_frames)
        window_right_writer.write_batch(window_right_annotated_frames)

        save(transformation_buffer, output_directory / 'buffer.json')
        Logger.info('Done!')

        Logger.info('Step complete')
