from pathlib import Path

import click
import torch
import viser
from child_lab_visualization import show_pointcloud_and_camera_poses
from serde.json import from_json
from serde.yaml import from_yaml
from transformation_buffer.buffer import Buffer
from video_io.calibration import Calibration
from video_io.reader import Reader

from ..workspace import Workspace, WorkspaceModelError


@click.command('visualize')
@click.argument('workspace_root', type=Path, metavar='<workspace>')
@click.argument('video_name', type=str, metavar='<video>')
def visualize(
    workspace_root: Path,
    video_name: str,
) -> None:
    """
    Display a point cloud for <video> from <workspace> in a local Viser server.
    """

    workspace = Workspace.in_directory(workspace_root)
    points = workspace.output / 'points' / video_name
    buffer_location = workspace.transformation / 'buffer.json'

    if not points.is_dir():
        raise WorkspaceModelError(
            f'{points} should contain serialized point clouds for video {video_name}'
        )

    if not buffer_location.is_file():
        raise WorkspaceModelError(
            f'{buffer_location} should contain the serialized transformation buffer'
        )

    for video in workspace.calibrated_videos():
        if video.name == video_name:
            break
    else:
        raise FileNotFoundError(f'Video {video_name} not found in {workspace.input}.')

    reader = Reader(video.location, torch.device('cpu'))
    calibration = from_yaml(Calibration, video.calibration.read_text())
    buffer = from_json(Buffer[str], buffer_location.read_text())

    points_files = sorted(points.glob('*'), key=lambda path: path.stem)

    server = viser.ViserServer()
    show_pointcloud_and_camera_poses(
        server,
        video_name,
        reader,
        points_files,
        calibration,
        buffer,
    )
    server.sleep_forever()
