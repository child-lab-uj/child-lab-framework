import sys
from .core.video import Reader, Writer, Perspective, Properties, Format
from .task import pose, face, gaze
from .task.visualization import Visualizer


def main() -> None:
    # Instantiate components:
    ceiling_reader = Reader(
        'dev/data/ultra_short/ceiling.mp4',
        perspective=Perspective.CEILING,
        batch_size=5
    )

    side_reader = Reader(
        'dev/data/short/window_left.mp4',
        perspective=Perspective.WINDOW_LEFT,
        batch_size=5
    )

    writer = Writer(
        'dev/output/ceiling_ultra_short_annotated.mp4',
        ceiling_reader.properties,
        output_format=Format.MP4
    )

    ceiling_pose_estimator = pose.Estimator(
        max_results=2,
        threshold=0.5
    )

    side_pose_estimator = pose.Estimator(
        max_results=2,
        threshold=0.5
    )

    face_estimator = face.Estimator(
        max_results=2,
        detection_threshold=0.1,
        tracking_threshold=0.1
    )

    gaze_estimator = gaze.Estimator()

    visualizer = Visualizer(confidence_threshold=0.5)

    # Open streams:
    ceiling_reader_thread = ceiling_reader.stream()
    side_reader_thread = ceiling_reader.stream()

    writer_thread = writer.stream()

    ceiling_pose_thread = ceiling_pose_estimator.stream()
    side_pose_thread = side_pose_estimator.stream()
    face_thread = face_estimator.stream()
    gaze_thread = gaze_estimator.stream()

    visualizer_thread = visualizer.stream()


    frames_count = 0
    fps = ceiling_reader.properties.fps

    while (
        (ceiling_frames := ceiling_reader_thread.send(None)) and
        (side_frames := side_reader_thread.send(None))
    ):
        ceiling_poses = ceiling_pose_thread.send(ceiling_frames)
        side_poses = side_pose_thread.send(side_frames)
        faces = face_thread.send(ceiling_frames)

        assert ceiling_poses is not None
        assert side_poses is not None
        assert faces is not None

        gazes = gaze_thread.send(gaze.Input(ceiling_poses, side_poses, faces))

        annotated_frames = visualizer_thread.send((ceiling_frames, ceiling_poses, gazes))
        writer_thread.send(annotated_frames)


        frames_count += len(ceiling_frames)
        seconds = frames_count / fps

        sys.stdout.write(f'\r{seconds = :.2f}')
        sys.stdout.flush()
