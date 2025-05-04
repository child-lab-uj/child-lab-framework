from pathlib import Path

import click
import cv2 as opencv
import torch
import viser
from child_lab_visualization.calibration_tuning.image import (
    show_image_with_calibration_controls,
)
from child_lab_visualization.calibration_tuning.point_cloud import (
    show_point_cloud_with_calibration_controls,
)
from depth_estimation.depth_pro import Configuration, DepthPro
from marker_detection.aruco import Dictionary, MarkerRigidModel
from video_io.frame import ArrayRgbFrame
from video_io.reader import Reader

from child_lab_cli.workspace.model import Workspace


@click.command('tune-calibration')
@click.argument('workspace-root', type=Path, metavar='<workspace>')
@click.argument('video-name', type=str, metavar='<video>')
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
    '--point-cloud',
    type=bool,
    is_flag=True,
    default=False,
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
    video_name: str,
    marker_dictionary: str,
    marker_size: float,
    point_cloud: bool,
    depth_pro_checkpoint: Path | None,
    device: str,
) -> None:
    workspace = Workspace.in_directory(workspace_root)

    video = next((v for v in workspace.videos() if v.name == video_name), None)
    if video is None:
        raise click.ClickException(
            f'Input video {video_name} not found in {workspace.input}'
        )

    reader = Reader(video.location)
    first_frame_tensor = reader.read()
    assert first_frame_tensor is not None
    first_frame: ArrayRgbFrame = first_frame_tensor.permute((1, 2, 0)).numpy()

    dictionary = Dictionary.parse(marker_dictionary)
    assert dictionary is not None

    marker_rigid_model = MarkerRigidModel(marker_size, 0.0, 0.005)

    server = viser.ViserServer()

    if point_cloud:
        assert depth_pro_checkpoint is not None
        depth_estimator = DepthPro(
            Configuration(depth_pro_checkpoint),
            torch.device(device),
            torch.half,
        )
        show_point_cloud_with_calibration_controls(
            server,
            first_frame_tensor.to(torch.device(device)),
            depth_estimator,
            marker_rigid_model,
            dictionary,
            opencv.aruco.DetectorParameters(),
        )
    else:
        show_image_with_calibration_controls(
            server,
            first_frame,
            marker_rigid_model,
            dictionary,
            opencv.aruco.DetectorParameters(),
        )

    server.sleep_forever()
