from pathlib import Path

import click
import torch
import viser
from child_lab_data.io.point_cloud import Reader as PointCloudReader
from child_lab_data.io.result_tensor import Reader as ResultReader
from child_lab_visualization.camera import show_camera
from child_lab_visualization.marker import show_marker
from child_lab_visualization.point_cloud import show_point_cloud
from child_lab_visualization.vpc_results import show_vpc_results
from serde.json import from_json
from serde.yaml import from_yaml
from transformation_buffer.buffer import Buffer
from video_io.calibration import Calibration
from video_io.reader import Reader as VideoReader
from vpc import gaze, pose

from ..workspace import Workspace, WorkspaceModelError


@click.command('visualize')
@click.argument('workspace_root', type=Path, metavar='<workspace>')
@click.argument('video_name', type=str, metavar='<video>')
def visualize(
    workspace_root: Path,
    video_name: str,
) -> None:
    """
    Display point clouds, cameras, markers and VPC results available in <workspace>
    in a local Viser server, in world coordinates defined by <video>.
    """

    workspace = Workspace.in_directory(workspace_root)
    buffer_location = workspace.transformation / 'buffer.json'

    if not buffer_location.is_file():
        raise WorkspaceModelError(
            f'{buffer_location} should contain the serialized transformation buffer'
        )

    transformation_buffer = from_json(Buffer[str], buffer_location.read_text())

    server = viser.ViserServer()

    frames = transformation_buffer.frames_visible_from(video_name)
    frames.append(video_name)

    for frame_name in frames:
        transformation = transformation_buffer[(frame_name, video_name)]
        assert transformation is not None

        if 'marker' in frame_name:
            show_marker(server, frame_name, transformation)
            continue

        video = workspace.calibrated_videos().find(lambda video: video.name == frame_name)
        if video is None:
            continue

        video_reader = VideoReader(video.location, torch.device('cpu'))
        calibration = from_yaml(Calibration, video.calibration.read_text())

        show_camera(
            server,
            frame_name,
            video_reader.metadata,
            calibration,
            transformation,
        )

        point_cloud_location = workspace.output / 'points' / frame_name
        pose_location = workspace.output / 'analysis' / 'pose' / frame_name
        gaze_location = workspace.output / 'analysis' / 'gaze' / frame_name

        if point_cloud_location.is_dir():
            point_cloud_reader = PointCloudReader(point_cloud_location)
            show_point_cloud(
                server,
                frame_name,
                video_reader,
                point_cloud_reader,
                transformation,
            )

        if pose_location.is_dir() and gaze_location.is_dir():
            pose_reader = ResultReader(
                pose.Result3d,
                workspace.output / 'analysis' / 'pose' / frame_name,
            )
            gaze_reader = ResultReader(
                gaze.Result3d,
                workspace.output / 'analysis' / 'gaze' / frame_name,
            )
            show_vpc_results(
                server,
                frame_name,
                transformation,
                pose_reader,
                gaze_reader,
            )

    server.sleep_forever()
