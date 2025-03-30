from pathlib import Path

import click
from child_lab_procedures.estimate_transformations import (
    Configuration,
    Procedure,
    VideoIoContext,
)
from marker_detection.aruco import (
    Dictionary,
    RigidModel,
    VisualizationContext,
)
from serde.json import to_json
from serde.yaml import from_yaml
from tqdm import trange
from transformation_buffer.rigid_model import Cube
from video_io.calibration import Calibration
from video_io.reader import Reader
from video_io.visualizer import Visualizer
from video_io.writer import Writer

from child_lab_cli.workspace.model import Workspace


@click.command('estimate-transformations')
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
    '--visualize',
    type=bool,
    is_flag=True,
    default=False,
    help='Produce videos with visualizations',
)
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
    help='File containing a serialized transformation buffer to load and place new transformations in',
)
@click.option(
    '--skip',
    type=int,
    required=False,
    help='Seconds of videos to skip at the beginning',
)
def estimate_transformations(
    workspace_root: Path,
    video_names: list[str],
    marker_dictionary: str,
    marker_size: float,
    visualize: bool,
    device: str | None,
    checkpoint: Path | None,
    skip: int | None,
) -> None:
    """
    Estimate mutual poses of cameras using <videos> from <workspace>
    by detecting ArUco markers of <size>, from <dictionary>
    and save them as a JSON-serialized transformation buffer.
    """

    workspace = Workspace.in_directory(workspace_root)

    buffer_destination = workspace.transformation / 'buffer.json'

    video_output = workspace.output / 'transformation'
    video_output.mkdir(exist_ok=True)

    video_io_contexts: list[VideoIoContext] = []
    calibrated_videos = workspace.calibrated_videos()

    for video_name in video_names:
        video = next((v for v in calibrated_videos if v.name == video_name), None)

        if video is None:
            raise click.ClickException(
                f'Input video {video_name} not found in {workspace.input}'
            )

        assert video.calibration.is_file()

        calibration = from_yaml(Calibration, video.calibration.read_text())

        reader = Reader(video.location)

        writer = (
            Writer(
                (video_output / video.name).with_suffix('.mp4'),
                reader.metadata,
                Visualizer[VisualizationContext](
                    {
                        'intrinsics': calibration.intrinsics_matrix().numpy(),
                        'marker_draw_masks': True,
                        'marker_draw_ids': True,
                        'marker_draw_axes': True,
                        'marker_draw_angles': True,
                        'marker_mask_color': (0.0, 1.0, 0.0, 1.0),
                        'marker_axis_length': 100,
                        'marker_axis_thickness': 1,
                    }
                ),
            )
            if visualize
            else None
        )

        context = VideoIoContext(
            video.name,
            calibration,
            reader,
            writer,
        )
        video_io_contexts.append(context)

    dictionary = Dictionary.parse(marker_dictionary)
    assert dictionary is not None

    configuration = Configuration(
        RigidModel(marker_size, 1.0),
        dictionary,
        arudice=DEFAULT_ARUDICE,
    )

    procedure = Procedure(configuration, video_io_contexts)
    progress_bar = trange(
        procedure.length_estimate(),
        desc='Estimating transformations...',
    )

    result = procedure.run(lambda: progress_bar.update())

    match result:
        case None:
            raise click.ClickException('Procedure has diverged')

        case buffer:
            buffer_destination.touch()
            buffer_destination.write_text(to_json(buffer))
            click.echo('Done!')


DEFAULT_ARUDICE = [
    Cube[str](
        50.0,
        ('marker_42', 'marker_43', 'marker_44', 'marker_45', 'marker_46', 'marker_47'),
    )
]
