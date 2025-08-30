from pathlib import Path

import click
import torch
from child_lab_data.io.point_cloud import Writer as PointCloudWriter
from child_lab_procedures.oak_d_capture import IoContext, Procedure
from rich.progress import Progress
from video_io import Writer as VideoWriter
from video_io.metadata import Metadata
from web_camera.oak_d import Reader as CameraReader

from .workspace import Workspace


@click.command('record')
@click.argument('workspace-root', type=Path, metavar='<workspace>')
@click.argument('camera-ip', type=str, metavar='<camera-ip>')
@click.option(
    '--name',
    type=str,
    required=False,
    help='Name of the output video',
)
@click.option(
    '--fps',
    type=int,
    default=30,
    help='Framerate to record with',
)
@click.option(
    '--device',
    type=str,
    default='cpu',
    help='Torch device to use for tensor computations',
)
def record(
    workspace_root: Path,
    camera_ip: str,
    name: str | None,
    fps: int,
    device: str,
) -> None:
    torch_device = torch.device(device)

    workspace = Workspace.in_directory(workspace_root)

    video_name = name or 'recording'  # TODO: add timestamp

    # Save the video in the input directory.
    video_destination = workspace.input / f'{video_name}.mp4'
    video_metadata = Metadata(float(fps), 0, 1920, 1080)
    video_writer = VideoWriter(video_destination, video_metadata)

    point_cloud_output = workspace.output / 'points' / video_name
    point_cloud_output.mkdir(exist_ok=True)
    point_cloud_writer = PointCloudWriter(point_cloud_output)

    # Keep the camera reader initialization as late as possible since there's no buffering.
    camera_reader = CameraReader(camera_ip, fps, device=torch_device)
    properties = camera_reader.runtime_properties()

    depth_map_output = workspace.output / 'depth' / f'{video_name}.mp4'
    depth_map_metadata = Metadata(
        float(fps),
        0,
        height=properties.stereo_camera_resolution[0],
        width=properties.stereo_camera_resolution[1],
    )
    depth_map_writer = VideoWriter(depth_map_output, depth_map_metadata)

    io_context = IoContext(
        properties.color_camera_calibration,
        camera_reader,
        video_writer,
        depth_map_writer,
        point_cloud_writer,
    )

    with Progress() as progress:
        assert isinstance(progress, Progress)

        recording_task = progress.add_task('[bold red]Recording...', total=None)
        frames_captured = 0
        average_fps = fps  # TODO: estimate properly

        def callback() -> None:
            nonlocal frames_captured
            frames_captured += 1
            progress.update(
                recording_task,
                advance=1,
                description=progress_description(frames_captured, average_fps),
            )

        Procedure(io_context).run(callback)


def progress_description(frames_captured: int, fps: float) -> str:
    return f'[bold red]Recording...[/bold red] [bold]{frames_captured}[/bold] frames captured, avg. [bold]{fps:.2f} fps[/bold]'
