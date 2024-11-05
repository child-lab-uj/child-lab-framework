from contextlib import ContextDecorator
from pathlib import Path
from types import TracebackType
from typing import Never, Self

import click
import torch

from .._procedure import calibrate as calibration_procedure
from .._procedure import demo_sequential
from .._procedure import estimate_transformations as transformation_procedure
from ..core.detection import chessboard, marker
from ..core.file import load, save
from ..core.video import Calibration, Input

# NOTE: This CLI is made for development purposes. It may not be a part of the final library.


class click_trap(ContextDecorator):
    def __init__(self) -> None:
        super().__init__()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _exception_kind: type | None,
        exception: Exception | None,
        _traceback: TracebackType | None,
        **_,
    ) -> bool | Never:
        if exception is not None:
            raise click.ClickException(str(exception))

        return False


@click.command('calibrate')
@click.argument('source', type=Path)
@click.argument('destination', type=Path)
@click.option('--square-size', type=float, help='Board square size in centimeters')
@click.option(
    '--inner-board-corners',
    nargs=2,
    type=int,
    help="Number of chessboard's inner corners in rows and columns that calibration algorithm should locate",
)
@click.option(
    '--skip',
    type=int,
    help='Number of source frames to skip upon each successful read and computation',
)
@click_trap()
def calibrate(
    source: Path,
    destination: Path,
    square_size: float,
    inner_board_corners: tuple[int, int],
    skip: int,
) -> None:
    click.echo(f'Calibrating camera from {source}...')

    result = calibration_procedure.run(
        Input('calibration', source, None),
        chessboard.Properties(square_size, *inner_board_corners),
        skip,
    )

    click.echo(f'Calibration complete! Estimated parameters:\n{result}')
    click.echo(f'Saving to {destination}...')

    save(result, destination)


@click.command('estimate-transformations')
@click.argument('workspace', type=Path)
@click.argument('videos', type=Path, nargs=-1)
@click.option('--marker-dictionary', type=str, help='Dictionary to detect markers from')
@click.option('--marker-size', type=float, help='Marker size in centimeters')
@click.option(
    '--device',
    type=str,
    required=False,
    help='Torch device to use for tensor computations',
)
@click.option(
    '--checkpoint',
    type=Path,
    required=False,
    help='File containing serialized Buffer to load and place new transformations in',
)
@click_trap()
def estimate_transformations(
    workspace: Path,
    videos: list[Path],
    marker_dictionary: str,
    marker_size: float,
    device: str | None,
    checkpoint: Path | None,
) -> None:
    video_dir = workspace / 'input'
    calibration_dir = workspace / 'calibration'
    destination = workspace / 'buffer.json'

    if not workspace.is_dir():
        raise ValueError(f'{workspace} is not valid workspace directory')

    if not video_dir.is_dir():
        raise ValueError(f'{video_dir} is not valid video directory')

    if not calibration_dir.is_dir():
        raise ValueError(f'{calibration_dir} is not valid calibration directory')

    device_handle = torch.device(device or 'cpu')
    model = marker.RigidModel(marker_size, 0.0)
    dictionary = marker.Dictionary.parse(marker_dictionary)

    if dictionary is None:
        raise click.ClickException(f'Unrecognized dictionary name: "{marker_dictionary}"')

    config = transformation_procedure.Config(model, dictionary)

    video_names = [path.name for path in videos]
    video_full_paths = [video_dir / video for video in videos]

    calibrations = [
        load(Calibration, calibration_dir / (name + '.yml')) for name in video_names
    ]

    inputs = [
        transformation_procedure.Input(name, video, calibration)
        for name, video, calibration in zip(video_names, video_full_paths, calibrations)
    ]

    click.echo('Estimating transformations...')

    result = transformation_procedure.run(inputs, config, device_handle)

    click.echo(f'Estimation complete! Estimated transformations:\n{result}')
    click.echo(f'Saving results to {destination}...')

    save(result, destination)


@click.command('process')
@click.argument('workspace', type=Path)
@click.argument('videos', type=Path, nargs=-1)
@click.option(
    '--device',
    type=str,
    required=False,
    help='Torch device to use for tensor computations',
)
# @click_trap()
def process(
    workspace: Path,
    videos: list[Path],
    device: str | None,
) -> None:
    video_dir = workspace / 'input'
    calibration_dir = workspace / 'calibration'
    destination = workspace / 'output'

    if not workspace.is_dir():
        raise ValueError(f'{workspace} is not valid workspace directory')

    if not video_dir.is_dir():
        raise ValueError(f'{video_dir} is not valid video directory')

    if not calibration_dir.is_dir():
        raise ValueError(f'{calibration_dir} is not valid calibration directory')

    if not destination.is_dir():
        raise ValueError(f'{destination} is not valid destination directory')

    device_handle = torch.device(device or 'cpu')

    video_names = [video.stem for video in videos]
    video_full_paths = [video_dir / video for video in videos]

    calibrations = [
        load(Calibration, calibration_dir / (name + '.yml')) for name in video_names
    ]

    inputs = [
        Input(name, video, calibration)
        for name, video, calibration in zip(video_names, video_full_paths, calibrations)
    ]

    click.echo('Proceeding with video analysis...')

    demo_sequential.main(inputs, device_handle, destination)  # type: ignore

    click.echo('Done!')


@click.group('child-lab')
def cli() -> None: ...


cli.add_command(calibrate)
cli.add_command(estimate_transformations)
cli.add_command(process)
