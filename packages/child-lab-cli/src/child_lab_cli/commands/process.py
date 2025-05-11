import typing
from pathlib import Path

import click
import torch
from child_lab_data.io.point_cloud import Reader as PointCloudReader
from child_lab_data.io.result_tensor import Writer as ResultWriter
from child_lab_procedures.process import (
    Configuration,
    Procedure,
    VideoIoContext,
    VisualizationContext,
)
from serde.yaml import from_yaml
from transformation_buffer import Buffer
from video_io.calibration import Calibration
from video_io.reader import Reader
from video_io.visualizer import Visualizer
from video_io.writer import Writer
from vpc import gaze, pose

from child_lab_cli.workspace.model import CalibratedVideo, Workspace


@click.command('process')
@click.argument('workspace-root', type=Path, metavar='<workspace>')
@click.argument('video_names', type=str, nargs=-1, metavar='<videos>')
@click.option(
    '--batch-size',
    type=int,
    default=16,
    help='Number of frames to process as a single batch',
)
@click.option(
    '--skip',
    type=int,
    required=False,
    help='Seconds of videos to skip at the beginning',
)
@click.option(
    '--device',
    type=str,
    default='cpu',
    help='Torch device to use for tensor computations',
)
def process(
    workspace_root: Path,
    video_names: list[str],
    batch_size: int,
    skip: int,
    device: str,
) -> None:
    torch_device = torch.device(device)

    workspace = Workspace.in_directory(workspace_root)
    output_directory = workspace.output / 'analysis'
    output_directory.mkdir(exist_ok=True)

    transformation_buffer: Buffer[str] | None = typing.cast(
        Buffer[str] | None,
        workspace.transformation_buffer(),
    )
    if transformation_buffer is None:
        # TODO: add expected location.
        raise FileNotFoundError('Transformation buffer not found in workspace')

    available_videos = workspace.calibrated_videos()
    videos: list[CalibratedVideo] = []
    calibrations: list[Calibration] = []

    for name in video_names:
        video = available_videos.find(lambda v: v.name == name)
        if video is None:
            raise FileNotFoundError(f'Video {name} not found in {workspace.input}.')

        videos.append(video)

        calibration = from_yaml(Calibration, video.calibration.read_text())
        calibrations.append(calibration)

    configuration = Configuration(
        torch_device,
        batch_size,
        max_detections=2,
        yolo_checkpoint=Path('model/yolov11x-pose.pt'),
        mini_face_models_directory=Path('model/model'),
        face_confidence_threshold=0.5,
        face_suppression_threshold=0.1,
    )

    visualization_context: VisualizationContext = {
        'pose_draw_bounding_boxes': True,
        'pose_bounding_box_color': (0, 0, 255),
        'pose_bounding_box_thickness': 2,
        'pose_bounding_box_min_confidence': 0.5,
        'pose_draw_keypoints': True,
        'pose_bone_color': (255, 0, 0),
        'pose_bone_thickness': 2,
        'pose_keypoint_color': (0, 255, 0),
        'pose_keypoint_radius': 3,
        'pose_keypoint_min_confidence': 0.5,
        'face_draw_bounding_boxes': True,
        'face_bounding_box_color': (0, 0, 255),
        'face_bounding_box_thickness': 2,
        'face_draw_confidence': False,
        'gaze_draw_lines': True,
        'gaze_line_color': (0, 255, 0),
        'gaze_line_length': 100.0,
        'gaze_line_thickness': 3,
    }

    io_contexts: list[VideoIoContext] = []

    for video, calibration in zip(videos, calibrations):
        video_reader = Reader(video.location, torch_device)
        visualizer = Visualizer(visualization_context)
        video_writer = Writer(
            output_directory / video.location.name,
            video_reader.metadata,
            visualizer,
        )

        point_cloud_reader = PointCloudReader(workspace.output / 'points' / video.name)

        pose_output = output_directory / 'pose' / video.location.stem
        pose_output.mkdir(parents=True, exist_ok=True)

        gaze_output = output_directory / 'gaze' / video.location.stem
        gaze_output.mkdir(parents=True, exist_ok=True)

        pose_writer = ResultWriter(pose.Result3d, pose_output)
        gaze_writer = ResultWriter(gaze.Result3d, gaze_output)

        context = VideoIoContext(
            video.location.stem,
            calibration,
            video_reader,
            point_cloud_reader,
            video_writer,
            pose_writer,
            gaze_writer,
        )
        io_contexts.append(context)

    procedure = Procedure(configuration, io_contexts, transformation_buffer)
    procedure.run()
