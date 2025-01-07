import shutil
from datetime import datetime
from pathlib import Path
from time import strptime

import click

from child_lab_framework.core.video import (
    SUPPORTED_SUFFIXES,
    Format,
    Input,
    Reader,
    Writer,
)
from child_lab_framework.core.workspace import Workspace

DEVELOPMENT_DIRECTORY = Path(__file__).parent.parent / 'dev'
DATA_LOCATION = DEVELOPMENT_DIRECTORY / 'data'
CALIBRATION_PRESETS_LOCATION = DEVELOPMENT_DIRECTORY / 'calibration'
TRANSFORMATION_PRESETS_LOCATION = DEVELOPMENT_DIRECTORY / 'transformation'
WORKSPACES_LOCATION = DEVELOPMENT_DIRECTORY / 'result'


@click.group
def workspace() -> None:
    """
    Manage workspaces
    """


@click.group
def video() -> None:
    """
    Modify input videos
    """


@workspace.command
@click.argument('name', type=str)
@click.option(
    '--data',
    '-d',
    required=False,
    type=Path,
    help='Path to a directory containing .avi or .mp4 movies to copy as a workspace input data, relative to the development directory',
)
@click.option(
    '--calibration',
    '-c',
    required=False,
    type=str,
    help='Name of the calibration preset to use in the workspace',
)
@click.option(
    '--transformation',
    '-t',
    required=False,
    type=str,
    help='Name of the transformation preset to use in the workspace',
)
def new(
    name: str,
    data: Path | None,
    calibration: str | None,
    transformation: str | None,
) -> None:
    """
    Create a new workspace
    """

    workspace_root = WORKSPACES_LOCATION / name

    if workspace_root.exists():
        raise click.ClickException(
            f'Cannot create workspace "{name}" - file with this name already exists in {WORKSPACES_LOCATION}'
        )

    if data is not None and not data.is_dir():
        raise click.ClickException(f'Cannot find data location "{data}"')

    if calibration is not None:
        calibration_preset_location = CALIBRATION_PRESETS_LOCATION / calibration

        if not calibration_preset_location.is_dir():
            raise click.ClickException(
                f'Cannot find calibration preset "{calibration_preset_location}"'
            )
    else:
        calibration_preset_location = None

    if transformation is not None:
        transformation_preset_location = CALIBRATION_PRESETS_LOCATION / transformation

        if not transformation_preset_location.is_dir():
            raise click.ClickException(
                f'Cannot find transformation preset "{transformation_preset_location}"'
            )
    else:
        transformation_preset_location = None

    input = workspace_root / 'input'
    output = workspace_root / 'output'
    archive = workspace_root / 'archive'
    workspace_calibration = workspace_root / 'calibration'
    workspace_transformation = workspace_root / 'transformation'

    for directory in (
        workspace_root,
        input,
        output,
        archive,
        workspace_calibration,
        workspace_transformation,
    ):
        directory.mkdir()

    if data is not None:
        for file in data.iterdir():
            if file.suffix not in SUPPORTED_SUFFIXES:
                continue

            shutil.copy(file, input)

    if calibration_preset_location is not None:
        shutil.copytree(calibration_preset_location, workspace_calibration)

    if transformation_preset_location is not None:
        shutil.copytree(transformation_preset_location, workspace_transformation)

    click.echo(f'Successfully initialized workspace "{name}" in {WORKSPACES_LOCATION}')


@workspace.command
@click.argument('workspace', type=str)
@click.option(
    '--name',
    '-n',
    required=False,
    type=str,
    help='A custom name of the archived directory',
)
def archive(
    workspace: str,
    name: str | None,
) -> None:
    """
    Archive the whole content of the workspace
    """

    workspace_root = WORKSPACES_LOCATION / workspace

    if not workspace_root.exists():
        raise click.ClickException(
            f'Cannot find workspace "{workspace}" in {WORKSPACES_LOCATION}'
        )

    archive = workspace_root / 'archive'

    if not archive.is_dir():
        raise click.ClickException(
            f'Workspace "{workspace}" doesn\'t contain the archive directory'
        )

    __archive_workspace_unchecked(workspace_root, archive, name)

    click.echo(f'Successfully archived workspace "{workspace}"')


@workspace.command
@click.argument('name', type=str)
def show(name: str) -> None:
    """
    Displays info about the workspace
    """

    root = WORKSPACES_LOCATION / name

    if not root.is_dir():
        raise click.ClickException(
            f'Cannot find workspace "{name}" in {WORKSPACES_LOCATION}'
        )

    print(Workspace.load(root))


@video.command
@click.argument('workspace_name', type=str)
@click.option(
    '--name',
    type=str,
    required=False,
    help='Name of the video to process. Cuts all available videos if not provided',
)
@click.option(
    '--time',
    type=str,
    nargs=2,
    required=True,
    help='Start and end of the cut video in format <minutes:seconds>',
)
@click.option(
    '--shape',
    nargs=2,
    type=int,
    required=False,
    default=(1080, 1920),
    help='Common height and width of the result videos',
)
@click.option(
    '--fps',
    type=int,
    required=False,
    default=50,
    help='Common FPS of the result videos',
)
@click.option(
    '--no-archive',
    type=bool,
    is_flag=True,
    default=False,
    help='Do not archive the workspace before modifying the videos',
)
def cut(
    workspace_name: str,
    name: str | None,
    time: tuple[str, str],
    shape: tuple[int, int],
    fps: int,
    no_archive: bool,
) -> None:
    """
    Cut all the input videos in the workspace to a common time and shape
    """

    workspace = Workspace.load(WORKSPACES_LOCATION / workspace_name)
    workspace_root = workspace.root

    if not no_archive:
        archive = workspace_root / 'archive'

        if not archive.is_dir():
            raise click.ClickException(
                f'Workspace "{workspace}" doesn\'t contain the archive directory'
            )

        __archive_workspace_unchecked(workspace_root, archive, None)

    input_directory = workspace_root / 'input'

    processed_input_directory = workspace_root / '__tmp_processed_input'
    processed_input_directory.mkdir()

    height, width = shape
    start = __parse_seconds(time[0])
    end = __parse_seconds(time[1])

    if start >= end:
        raise click.ClickException(
            f'Expected start time < end time, got {start = } and {end = }'
        )

    if name is not None:
        input_candidates = [input for input in workspace.inputs if input.name == name]

        if len(input_candidates) == 0:
            raise click.ClickException(f'Cannot find video "{name}" in {input_directory}')

        input = input_candidates[0]

        __cut_video_unchecked(
            input,
            processed_input_directory,
            height,
            width,
            fps,
            start,
            end,
        )

        input.source.unlink()

    else:
        for input in workspace.inputs:
            __cut_video_unchecked(
                input,
                processed_input_directory,
                height,
                width,
                fps,
                start,
                end,
            )

            input.source.unlink()

    for file in processed_input_directory.iterdir():
        shutil.copy(file, input_directory)

    shutil.rmtree(processed_input_directory)

    click.echo(f'Successfully cut input videos in workspace "{workspace_name}"')


def __parse_seconds(encoded_time: str) -> int:
    parsed = strptime(encoded_time, '%M:%S')
    return parsed.tm_sec + 60 * parsed.tm_min


def __cut_video_unchecked(
    input: Input,
    destination_directory: Path,
    height: int,
    width: int,
    fps: int,
    start: int,
    end: int,
) -> None:
    reader = Reader(input, height=height, width=width, fps=fps, batch_size=1)
    reader.read_skipping(start * fps)

    writer = Writer(
        destination_directory / (input.source.stem + '.mp4'),
        reader.properties,
        output_format=Format.MP4,
    )

    for _ in range((end - start) * fps):
        frame = reader.read()
        assert frame is not None
        writer.write(frame)


def __archive_workspace_unchecked(
    workspace_root: Path,
    archive: Path,
    entry_name: str | None,
) -> None:
    entry_name = (
        entry_name
        if entry_name is not None
        else ('archive_' + datetime.now().isoformat('_', timespec='seconds'))
    )

    archive_entry = archive / entry_name
    archive_entry.mkdir()  # safe to assume that such entry does not exist

    for item in workspace_root.iterdir():
        if item.name == 'archive':
            continue

        if item.is_file():
            shutil.copy(item, archive_entry)

        elif item.is_dir():
            destination = archive_entry / item.name
            shutil.copytree(item, destination)
