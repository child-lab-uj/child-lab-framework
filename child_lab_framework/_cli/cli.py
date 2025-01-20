from contextlib import ContextDecorator
from pathlib import Path
from types import TracebackType
from typing import Literal, Self

import click
import torch

from .._procedure import calibrate as calibration_procedure
from .._procedure import demo_sequential
from .._procedure import estimate_transformations as transformation_procedure
from ..core import transformation
from ..core.calibration import Calibration
from ..core.file import load, save
from ..core.video import Input
from ..task.camera.detection import chessboard, marker

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
        **_: object,
    ) -> Literal[False]:
        if exception is not None:
            raise click.ClickException(str(exception))

        return False


@click.command('calibrate')
@click.argument('workspace', type=Path)
@click.argument('videos', type=Path, nargs=-1)
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
    workspace: Path,
    videos: list[Path],
    square_size: float,
    inner_board_corners: tuple[int, int],
    skip: int,
) -> None:
    video_input_dir = workspace / 'input'
    calibration_output_dir = workspace / 'calibration'
    video_output_dir = workspace / 'output'

    if not workspace.is_dir():
        raise ValueError(f'{workspace} is not valid workspace directory')

    if not video_input_dir.is_dir():
        raise ValueError(f'{video_input_dir} is not valid video input directory')

    if not calibration_output_dir.is_dir():
        raise ValueError(
            f'{calibration_output_dir} is not valid calibration output directory'
        )

    if not video_output_dir.is_dir():
        raise ValueError(f'{calibration_output_dir} is not valid video output directory')

    for video in videos:
        click.echo(f'Calibrating camera from {video}...')

        video_input = video_input_dir / video
        video_output = video_output_dir / video
        calibration_output = calibration_output_dir / f'{video.stem}.yml'

        calibration = calibration_procedure.run(
            video_input,
            video_output,
            chessboard.Properties(square_size, *inner_board_corners),
            skip,
        )

        click.echo(f'Calibration complete! Estimated parameters:\n{calibration}')
        click.echo(f'Saving results to {calibration_output}...')
        click.echo('')

        save(calibration, calibration_output)


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
@click.option(
    '--skip',
    type=int,
    required=False,
    help='Seconds of videos to skip at the beginning',
)
@click_trap()
def estimate_transformations(
    workspace: Path,
    videos: list[Path],
    marker_dictionary: str,
    marker_size: float,
    device: str | None,
    checkpoint: Path | None,
    skip: int | None,
) -> None:
    video_input_dir = workspace / 'input'
    video_output_dir = workspace / 'output'
    calibration_input_dir = workspace / 'calibration'
    transformation_output_dir = workspace / 'transformation'
    transformation_output = transformation_output_dir / 'buffer.json'

    if not workspace.is_dir():
        raise ValueError(f'{workspace} is not valid workspace directory')

    if not video_input_dir.is_dir():
        raise ValueError(f'{video_input_dir} is not valid video input directory')

    if not video_output_dir.is_dir():
        raise ValueError(f'{video_output_dir} is not valid video output directory')

    if not calibration_input_dir.is_dir():
        raise ValueError(
            f'{calibration_input_dir} is not valid calibration input directory'
        )

    if not transformation_output_dir.is_dir():
        raise ValueError(
            f'{calibration_input_dir} is not valid transformation output directory'
        )

    device_handle = torch.device(device or 'cpu')
    model = marker.RigidModel(marker_size, 0.0)
    dictionary = marker.Dictionary.parse(marker_dictionary)

    if dictionary is None:
        raise click.ClickException(f'Unrecognized dictionary name: "{marker_dictionary}"')

    configuration = transformation_procedure.Configuration(model, dictionary)

    calibrations = [
        load(Calibration, calibration_input_dir / (video.stem + '.yml'))
        for video in videos
    ]

    click.echo('Estimating transformations...')

    buffer = transformation_procedure.run(
        [video_input_dir / video for video in videos],
        [video_output_dir / video for video in videos],
        calibrations,
        skip if skip is not None else 0,
        configuration,
        device_handle,
    )

    click.echo(f'Estimation complete! Estimated transformations:\n{buffer}')
    click.echo(f'Saving results to {transformation_output}...')

    save(buffer, transformation_output)


@click.command('process')
@click.argument('workspace', type=Path)
@click.argument('videos', type=Path, nargs=-1)
@click.option(
    '--device',
    type=str,
    required=False,
    help='Torch device to use for tensor computations',
)
@click.option(
    '--skip',
    type=int,
    required=False,
    help='Seconds of videos to skip at the beginning',
)
@click.option(
    '--dynamic-transformations',
    type=bool,
    is_flag=True,
    default=False,
    help='Compute camera transformations on the fly, using heuristic algorithms',
)
# @click_trap()
def process(
    workspace: Path,
    videos: list[Path],
    device: str | None,
    skip: int | None,
    dynamic_transformations: bool,
) -> None:
    video_dir = workspace / 'input'
    calibration_dir = workspace / 'calibration'
    destination = workspace / 'output'
    transformation_dir = workspace / 'transformation'
    transformation_buffer_location = transformation_dir / 'buffer.json'

    if not workspace.is_dir():
        raise ValueError(f'{workspace} is not valid workspace directory')

    if not video_dir.is_dir():
        raise ValueError(f'{video_dir} is not valid video directory')

    if not calibration_dir.is_dir():
        raise ValueError(f'{calibration_dir} is not valid calibration directory')

    if not destination.is_dir():
        raise ValueError(f'{destination} is not valid destination directory')

    if not dynamic_transformations and not transformation_buffer_location.is_file():
        raise ValueError(
            f"""
            Computation requires a camera model. Please either:
                * Provide a buffer with static transformations at {transformation_buffer_location.absolute()}
                * Use --dynamic-transformation flag to compute them on the fly
            """
        )

    transformation_buffer: transformation.Buffer[str] | None = (
        load(transformation.Buffer, transformation_buffer_location)
        if transformation_buffer_location.is_file()
        else None
    )

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

    click.echo(f'Processing {"video" if len(videos) == 1 else "videos"}...')

    demo_sequential.main(
        inputs,  # type: ignore
        device_handle,
        destination,
        skip,
        transformation_buffer,
        dynamic_transformations,
    )

    click.echo('Done!')


@click.group('child-lab')
def cli() -> None: ...


cli.add_command(calibrate)
cli.add_command(estimate_transformations)
cli.add_command(process)
