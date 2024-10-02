from concurrent.futures import ThreadPoolExecutor

from .core.video import Reader, Writer, Perspective, Format
from .core.flow import Machinery
from .task import pose, face, gaze, depth, social_distance
from .task.visualization import Visualizer
from .task.camera.transformation import heuristic

BATCH_SIZE = 5


async def main() -> None:
    executor = ThreadPoolExecutor(max_workers=8)

    ceiling_reader = Reader(
        'dev/data/ultra_short/ceiling.mp4',
        perspective=Perspective.CEILING,
        batch_size=BATCH_SIZE,
    )

    window_left_reader = Reader(
        'dev/data/short/window_left.mp4',
        perspective=Perspective.WINDOW_LEFT,
        batch_size=BATCH_SIZE,
    )

    window_right_reader = Reader(
        'dev/data/short/window_right.mp4',
        perspective=Perspective.WINDOW_RIGHT,
        batch_size=BATCH_SIZE,
    )

    ceiling_depth_estimator = depth.Estimator(executor, inter_threads=3)
    window_left_depth_estimator = depth.Estimator(executor, inter_threads=3)
    window_right_depth_estimator = depth.Estimator(executor, inter_threads=3)

    ceiling_pose_estimator = pose.Estimator(executor, max_detections=2, threshold=0.5)

    window_left_pose_estimator = pose.Estimator(executor, max_detections=2, threshold=0.5)

    window_right_pose_estimator = pose.Estimator(
        executor, max_detections=2, threshold=0.5
    )

    window_left_to_ceiling_transformation_estimator = heuristic.Estimator(
        window_left_reader.properties, ceiling_reader.properties, executor
    )

    window_right_to_ceiling_transformation_estimator = heuristic.Estimator(
        window_right_reader.properties, ceiling_reader.properties, executor
    )

    window_left_face_estimator = face.Estimator(
        executor, max_results=2, detection_threshold=0.1, tracking_threshold=0.1
    )

    window_right_face_estimator = face.Estimator(
        executor, max_results=2, detection_threshold=0.1, tracking_threshold=0.1
    )

    gaze_estimator = gaze.Estimator(
        executor,
        ceiling_reader.properties,
        window_left_reader.properties,
        window_right_reader.properties,
    )

    social_distance_estimator = social_distance.Estimator(executor)
    social_distance_logger = social_distance.FileLogger('dev/output/distance.csv')

    visualizer = Visualizer(executor, confidence_threshold=0.5)

    writer = Writer(
        'dev/output/gaze_test.mp4', ceiling_reader.properties, output_format=Format.MP4
    )

    machinery = Machinery(
        [
            ('ceiling_reader', ceiling_reader),
            ('window_left_reader', window_left_reader),
            ('window_right_reader', window_right_reader),
            ('ceiling_depth', ceiling_depth_estimator, 'ceiling_reader'),
            ('window_left_depth', window_left_depth_estimator, 'window_left_reader'),
            ('window_right_depth', window_right_depth_estimator, 'window_right_reader'),
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
            ('window_left_face', window_left_face_estimator, 'window_left_reader'),
            ('window_right_face', window_right_face_estimator, 'window_right_reader'),
            (
                'gaze',
                gaze_estimator,
                (
                    'ceiling_pose',
                    'window_left_pose',
                    'window_right_pose',
                    'window_left_face',
                    'window_right_face',
                    'window_left_to_ceiling',
                    'window_right_to_ceiling',
                ),
            ),
            ('social_distance', social_distance_estimator, 'ceiling_pose'),
            ('social_distance_logger', social_distance_logger, 'social_distance'),
            ('visualizer', visualizer, ('ceiling_reader', 'ceiling_pose', 'gaze')),
            ('writer', writer, 'visualizer'),
        ]
    )

    await machinery.run()
