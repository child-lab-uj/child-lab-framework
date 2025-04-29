from pathlib import Path

import click
from child_lab_procedures.calibrate_camera import Configuration, Procedure, VideoIoContext
from marker_detection.chessboard import BoardProperties, VisualizationContext
from serde.yaml import to_yaml
from tqdm import trange
from video_io import Reader, Visualizer, Writer

from child_lab_cli.workspace.model import Workspace


@click.command('calibrate-camera', options_metavar='<options>')
@click.argument('workspace-root', type=Path, metavar='<workspace>')
@click.argument('video-name', type=str, metavar='<video>')
@click.option(
    '--square-size',
    type=float,
    required=True,
    help='Board square size in centimeters',
    metavar='<square-size>',
)
@click.option(
    '--inner-board-corners',
    nargs=2,
    type=int,
    required=True,
    help="Number of chessboard's inner corners in rows and columns",
    metavar='<inner-shape>',
)
@click.option(
    '--max-samples',
    type=int,
    required=False,
    help='Maximal number of board samples to collect',
)
@click.option(
    '--max-speed',
    type=float,
    default=float('inf'),
    required=False,
    help='Maximal speed the board can move with to be captured, in pixels per second',
)
@click.option(
    '--min-distance',
    type=float,
    default=0.3,
    required=False,
    help='Minimal distance between new observation and the previous observations to be captured',
)
def calibrate_camera(
    workspace_root: Path,
    video_name: str,
    square_size: float,
    inner_board_corners: tuple[int, int],
    max_samples: int | None,
    max_speed: float,
    min_distance: float,
) -> None:
    """
    Calibrate the camera using <video> from <workspace> by detecting
    inner corners of a chessboard of <inner-shape> with <square-size>.
    """

    workspace = Workspace.in_directory(workspace_root)

    video_output = workspace.output / 'calibration'
    video_output.mkdir(exist_ok=True)
    video_destination = (video_output / video_name).with_suffix('.mp4')

    calibration_destination = (workspace.calibration / video_name).with_suffix('.yml')

    video = next((v for v in workspace.videos() if v.name == video_name), None)
    if video is None:
        raise click.ClickException(
            f'Input video {video_name} not found in {workspace.input}'
        )

    reader = Reader(video.location)
    writer = Writer(
        video_destination,
        reader.metadata,
        Visualizer[VisualizationContext]({'chessboard_draw_corners': True}),
    )
    video_io_context = VideoIoContext(video_name, reader, writer)

    configuration = Configuration(
        BoardProperties(square_size, *inner_board_corners),
        max_samples,
        max_speed,
        min_distance,
    )

    procedure = Procedure(configuration, video_io_context)

    progress_bar = trange(
        procedure.length_estimate(),
        desc='Gathering samples for calibration...',
    )

    samples = procedure.run(lambda: progress_bar.update())
    match samples:
        case None:
            raise click.ClickException('Procedure has diverged')

        case samples:
            click.echo('Computing calibration...')
            result = samples.calibrate()
            calibration_destination.touch()
            calibration_destination.write_text(to_yaml(result.calibration))
