from collections.abc import Generator
from pathlib import Path

import click
import torch
import tqdm
from child_lab_data.io.point_cloud import Writer as PointCloudWriter
from depth_estimation import depth_pro
from serde.yaml import from_yaml
from video_io.calibration import Calibration
from video_io.reader import Reader

from ..workspace import Workspace


@click.command('generate-pointcloud')
@click.argument('workspace_root', type=Path, metavar='<workspace>')
@click.argument('video_name', type=str, metavar='<video>')
@click.option(
    '--checkpoint',
    type=Path,
    required=True,
    help='DepthPro checkpoint to build the depth estimator from',
)
@click.option(
    '--batch-size',
    type=int,
    default=16,
    required=False,
    help='Number of frames to process as a single batch',
)
@click.option(
    '--device',
    type=str,
    default='cpu',
    required=False,
    help='Torch device to use for tensor computations',
)
def generate_pointcloud(
    workspace_root: Path,
    video_name: str,
    checkpoint: Path,
    batch_size: int,
    device: str,
) -> None:
    """
    Generate a point cloud for <video> from <workspace> by estimating the depth.
    """

    workspace = Workspace.in_directory(workspace_root)

    video = workspace.calibrated_videos().find(lambda video: video.name == video_name)
    if video is None:
        raise FileNotFoundError(f'Video {video_name} not found in {workspace.input}.')

    output = workspace.output / 'points' / video_name
    output.mkdir(exist_ok=True)
    point_cloud_writer = PointCloudWriter(output)

    calibration = from_yaml(Calibration, video.calibration.read_text())
    fx = calibration.focal_length[0]

    main_device = torch.device(device)

    reader = Reader(video.location, torch.device('cpu'))

    depth_estimator = depth_pro.DepthPro(
        depth_pro.Configuration(checkpoint=checkpoint),
        main_device,
        torch.half,
    )

    fx = calibration.focal_length[0]
    fx = fx * reader.metadata.width / depth_estimator.input_image_size

    click.echo('Depth estimator created!')

    def batched_frames(
        reader: Reader,
        batch_size: int,
    ) -> Generator[torch.Tensor, None, None]:
        while (frames := reader.read_batch(batch_size)) is not None:
            yield frames

    progress_bar = tqdm.tqdm(
        range(0, reader.metadata.frames, batch_size),
        desc='Processing batches of frames',
    )

    for frame_batch in batched_frames(reader, batch_size):
        # TODO: Process whole batches of frames.
        depths = [
            depth_estimator.predict(frame.to(main_device), fx).depth.cpu()
            for frame in frame_batch.unbind()
        ]

        for depth in depths:
            point_cloud = calibration.unproject_depth(depth)
            point_cloud_writer.write(point_cloud)

        progress_bar.update()

    click.echo('Done!')
