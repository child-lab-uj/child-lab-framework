import os
from concurrent.futures import ThreadPoolExecutor

import torch

from .core.flow import Machinery
from .core.video import Format, Perspective, Reader, Writer
from .task import face, gaze, pose
from .task.visualization import Visualizer

BATCH_SIZE = 32


async def main() -> None:
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

    # ceiling_depth_estimator = depth.Estimator(
    #     executor, gpu, input=ceiling_reader.properties
    # )

    # window_left_depth_estimator = depth.Estimator(
    #     executor, gpu, input=window_left_reader.properties
    # )

    # window_right_depth_estimator = depth.Estimator(
    #     executor, gpu, input=window_right_reader.properties
    # )

    ceiling_pose_estimator = pose.Estimator(
        executor,
        gpu,
        input=ceiling_reader.properties,
        max_detections=2,
        threshold=0.5,
    )

    window_left_pose_estimator = pose.Estimator(
        executor,
        gpu,
        input=window_left_reader.properties,
        max_detections=2,
        threshold=0.5,
    )

    window_right_pose_estimator = pose.Estimator(
        executor,
        gpu,
        input=window_right_reader.properties,
        max_detections=2,
        threshold=0.5,
    )

    # window_left_to_ceiling_transformation_estimator = heuristic.Estimator(
    #     executor,
    #     window_left_reader.properties,
    #     ceiling_reader.properties,
    #     keypoint_threshold=0.5,
    # )

    # window_right_to_ceiling_transformation_estimator = heuristic.Estimator(
    #     executor,
    #     window_right_reader.properties,
    #     ceiling_reader.properties,
    #     keypoint_threshold=0.5,
    # )

    window_left_face_estimator = face.Estimator(
        executor, input=window_left_reader.properties, threshold=0.1
    )

    window_right_face_estimator = face.Estimator(
        executor, input=window_right_reader.properties, threshold=0.1
    )

    window_left_gaze_estimator = gaze.Estimator(
        executor,
        input=window_left_reader.properties,
    )

    window_right_gaze_estimator = gaze.Estimator(
        executor,
        input=window_right_reader.properties,
    )

    # gaze_projector = gaze.ceiling_projection.Estimator(
    #     executor,
    #     ceiling_reader.properties,
    #     window_left_reader.properties,
    #     window_right_reader.properties,
    # )

    # social_distance_estimator = social_distance.Estimator(executor)
    # social_distance_logger = social_distance.FileLogger('dev/output/distance.csv')

    window_left_visualizer = Visualizer(
        executor, properties=window_left_reader.properties, confidence_threshold=0.5
    )

    window_right_visualizer = Visualizer(
        executor, properties=window_right_reader.properties, confidence_threshold=0.5
    )

    window_left_writer = Writer(
        'dev/output/face_window_left.mp4',
        window_left_reader.properties,
        output_format=Format.MP4,
    )

    window_right_writer = Writer(
        'dev/output/face_window_right.mp4',
        window_right_reader.properties,
        output_format=Format.MP4,
    )

    machinery = Machinery(
        [
            ('ceiling_reader', ceiling_reader),
            ('window_left_reader', window_left_reader),
            ('window_right_reader', window_right_reader),
            # ('ceiling_depth', ceiling_depth_estimator, 'ceiling_reader'),
            # ('window_left_depth', window_left_depth_estimator, 'window_left_reader'),
            # ('window_right_depth', window_right_depth_estimator, 'window_right_reader'),
            ('ceiling_pose', ceiling_pose_estimator, 'ceiling_reader'),
            ('window_left_pose', window_left_pose_estimator, 'window_left_reader'),
            ('window_right_pose', window_right_pose_estimator, 'window_right_reader'),
            # (
            #     'window_left_to_ceiling',
            #     window_left_to_ceiling_transformation_estimator,
            #     (
            #         'window_left_pose',
            #         'ceiling_pose',
            #         'window_left_depth',
            #         'ceiling_depth',
            #     ),
            # ),
            # (
            #     'window_right_to_ceiling',
            #     window_right_to_ceiling_transformation_estimator,
            #     (
            #         'window_right_pose',
            #         'ceiling_pose',
            #         'window_right_depth',
            #         'ceiling_depth',
            #     ),
            # ),
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
            # (
            #     'ceiling_gaze',
            #     gaze_projector,
            #     (
            #         'ceiling_pose',
            #         'window_left_gaze',
            #         'window_right_gaze',
            #         'window_left_to_ceiling',
            #         'window_right_to_ceiling',
            #     ),
            # ),
            # ('social_distance', social_distance_estimator, 'ceiling_pose'),
            # ('social_distance_logger', social_distance_logger, 'social_distance'),
            (
                'window_left_visualizer',
                window_left_visualizer,
                (
                    'window_left_reader',
                    'window_left_pose',
                    'window_left_face',
                ),
            ),
            (
                'window_right_visualizer',
                window_right_visualizer,
                (
                    'window_right_reader',
                    'window_right_pose',
                    'window_right_face',
                ),
            ),
            ('window_left_writer', window_left_writer, 'window_left_visualizer'),
            ('window_right_writer', window_right_writer, 'window_right_visualizer'),
        ]
    )

    await machinery.run()
