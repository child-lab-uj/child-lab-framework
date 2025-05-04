from collections.abc import Generator
from pathlib import Path

import click
import torch
import tqdm
from depth_estimation import depth_pro
from jaxtyping import UInt8
from serde.yaml import from_yaml
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, Resize
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

    if not output.is_dir():
        output.mkdir()

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

    for i, frames in enumerate(batched_frames(reader, batch_size)):
        depths = [
            depth_estimator.predict(frame.to(main_device), fx).depth.cpu()
            for frame in frames.unbind()
        ]

        perspective_points = torch.stack(
            [calibration.unproject_depth(depth) for depth in depths]
        )

        torch.save(perspective_points, output / f'points_{i}.pt')

        progress_bar.update()

    click.echo('Done!')


# TODO: Delete this from here
class DepthEstimator:
    device: torch.device
    model: depth_pro.DepthPro
    model_config: depth_pro.Configuration
    to_model: Compose

    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        self.device = device

        config = depth_pro.Configuration(checkpoint=checkpoint)
        self.model_config = config
        self.model = depth_pro.DepthPro(config, device, torch.half)

        self.to_model = Compose(  # type: ignore[no-untyped-call]
            [
                ConvertImageDtype(torch.half),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # type: ignore[no-untyped-call]
            ]
        )

    def predict(
        self,
        frame_batch: UInt8[torch.Tensor, 'batch 3 height width'],
        focal_length_x: float | None = None,
    ) -> torch.Tensor:
        *_, height, width = frame_batch.shape

        scaled_focal_length = (
            (focal_length_x * self.model.input_image_size / width)
            if focal_length_x is not None
            else None
        )

        result = self.model.predict(self.to_model(frame_batch), scaled_focal_length)

        return (  # type: ignore[no-any-return] # mypy infers Any here for some reason.
            Resize((height, width))  # type: ignore[no-untyped-call]
            .forward(result.depth)
            .to(torch.float32)
        )
