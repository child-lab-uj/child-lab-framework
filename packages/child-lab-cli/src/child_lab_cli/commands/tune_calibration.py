from pathlib import Path
from typing import Literal

import click
import cv2 as opencv
import torch
import viser
from child_lab_visualization.calibration_tuning.image import (
    show_image_with_calibration_controls,
)
from child_lab_visualization.calibration_tuning.point_cloud import (
    Input,
    show_point_cloud_with_calibration_controls,
)
from depth_estimation.depth_pro import Configuration, DepthPro
from marker_detection.aruco import Dictionary, MarkerRigidModel
from transformation_buffer.rigid_model import Cube
from video_io.reader import Reader as VideoReader

from child_lab_cli.workspace.model import Video, Workspace


class Mode(click.ParamType):
    name = 'mode'

    def convert(self, value, param, ctx) -> Literal['image', 'point-cloud']:
        if value == 'image':
            return 'image'
        if value == 'point-cloud':
            return 'point-cloud'

        self.fail(
            f'Unrecognized mode: "{value}". Possible values: "image", "point-cloud"',
            param,
            ctx,
        )


@click.command('tune-calibration')
@click.argument('workspace-root', type=Path, metavar='<workspace>')
@click.argument('video-names', type=str, nargs=-1, metavar='<videos>')
@click.option(
    '--marker-dictionary',
    type=str,
    help='Dictionary to detect markers from',
    metavar='<dictionary>',
)
@click.option(
    '--marker-size',
    type=float,
    help='Marker size in centimeters',
    metavar='<size>',
)
@click.option(
    '--mode',
    type=Mode(),
    help='Display 3D point-cloud with embedded ArUco markers',
)
@click.option(
    '--depth-pro-checkpoint',
    type=Path,
    required=False,
    help='Location of DepthPro weights (used only with --point-cloud)',
)
@click.option(
    '--device',
    type=str,
    default='cpu',
    help='Torch device to use for depth estimation (used only with --point-cloud)',
)
def tune_calibration(
    workspace_root: Path,
    video_names: str,
    marker_dictionary: str,
    marker_size: float,
    mode: Literal['image', 'point-cloud'],
    depth_pro_checkpoint: Path | None,
    device: str,
) -> None:
    workspace = Workspace.in_directory(workspace_root)

    dictionary = Dictionary.parse(marker_dictionary)
    assert dictionary is not None

    marker_rigid_model = MarkerRigidModel(marker_size, 0.0, 0.005)

    match mode:
        case 'image':
            if len(video_names) != 1:
                raise click.ClickException(
                    'Exactly one video is required in "image" mode'
                )

            video_name = video_names[0]

            video = workspace.videos().find(lambda video: video.name == video_name)
            if video is None:
                raise click.ClickException(
                    f'Input video {video_name} not found in {workspace.input}'
                )

            server = viser.ViserServer()
            handle_image_mode(server, video, marker_rigid_model, dictionary)

        case 'point-cloud':
            videos: list[Video] = []

            for video_name in video_names:
                video = workspace.videos().find(lambda video: video.name == video_name)
                if video is None:
                    raise click.ClickException(
                        f'Input video {video_name} not found in {workspace.input}'
                    )
                videos.append(video)

            if depth_pro_checkpoint is None:
                raise click.ClickException(
                    '--depth-pro-checkpoint is required in "point-cloud" mode'
                )

            depth_estimator = DepthPro(
                Configuration(depth_pro_checkpoint),
                torch.device(device),
                torch.half,
            )

            server = viser.ViserServer()
            handle_point_cloud_mode(
                server,
                videos,
                marker_rigid_model,
                dictionary,
                depth_estimator,
                torch.device(device),
            )

    server.sleep_forever()


def handle_image_mode(
    server: viser.ViserServer,
    video: Video,
    marker_rigid_model: MarkerRigidModel,
    marker_dictionary: Dictionary,
) -> None:
    reader = VideoReader(video.location)

    first_frame_tensor = reader.read()
    assert first_frame_tensor is not None
    first_frame = first_frame_tensor.permute((1, 2, 0)).numpy()

    show_image_with_calibration_controls(
        server,
        first_frame,
        marker_rigid_model,
        marker_dictionary,
        opencv.aruco.DetectorParameters(),
    )


def handle_point_cloud_mode(
    server: viser.ViserServer,
    videos: list[Video],
    marker_rigid_model: MarkerRigidModel,
    marker_dictionary: Dictionary,
    depth_estimator: DepthPro,
    device: torch.device,
) -> None:
    inputs: list[Input] = []

    for video in videos:
        reader = VideoReader(video.location)

        first_frame = reader.read()
        assert first_frame is not None
        first_frame = first_frame.to(torch.device(device))

        inputs.append(Input(video.name, video.location, first_frame))
        del reader  # make sure all videos are closes before proceeding to the next part.

    arudie = Cube[str](
        marker_rigid_model.square_size + 2 * marker_rigid_model.border,
        # TODO: Read the wall markers from external configuration
        ('marker_42', 'marker_43', 'marker_44', 'marker_45', 'marker_46', 'marker_47'),
    )

    show_point_cloud_with_calibration_controls(
        server,
        inputs,
        depth_estimator,
        marker_rigid_model,
        marker_dictionary,
        [arudie],
    )
