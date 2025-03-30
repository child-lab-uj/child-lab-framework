import shutil
from datetime import datetime
from pathlib import Path

import click

from child_lab_cli.workspace.model import Workspace


@click.group('workspace')
def workspace() -> None: ...


@workspace.command('new')
@click.argument('name', type=str)
@click.argument('location', type=Path)
@click.option(
    '--data',
    '-d',
    required=False,
    type=Path,
    help='Path to a directory containing .avi or .mp4 movies to copy as a workspace input data',
)
@click.option(
    '--calibration',
    '-c',
    required=False,
    type=Path,
    help='Path to a directory containing serialized camera intrinsics to use in the workspace',
)
@click.option(
    '--transformation',
    '-t',
    required=False,
    type=Path,
    help='Path to a directory containing transformation data to use in the workspacee',
)
def new(
    name: str,
    location: Path,
    data: Path | None,
    calibration: Path | None,
    transformation: Path | None,
) -> None:
    """
    Create a new workspace
    """

    if not location.is_dir():
        raise click.ClickException(
            f'Cannot create workspace at {location} - the directory does not exist'
        )

    if data is not None and not data.is_dir():
        raise click.ClickException(f'Cannot find data at {data}')

    if calibration is not None and not calibration.is_dir():
        raise click.ClickException(f'Cannot find calibration at {calibration}')

    if transformation is not None and not transformation.is_dir():
        raise click.ClickException(f'Cannot find transformation at {transformation}')

    workspace = Workspace.in_directory(location / name)

    if data is not None:
        shutil.copytree(data, workspace.input)

    if calibration is not None:
        shutil.copytree(calibration, workspace.calibration)

    if transformation is not None:
        shutil.copytree(transformation, workspace.transformation)

    click.echo(f'Successfully initialized workspace "{name}" at {workspace.root}')


@workspace.command('archive')
@click.argument('workspace_root', type=Path)
@click.option(
    '--name',
    '-n',
    required=False,
    type=str,
    help='A custom name of the archived directory',
)
def archive(
    workspace_root: Path,
    name: str | None,
) -> None:
    """
    Archive the whole content of the workspace
    """

    workspace = Workspace.in_directory(workspace_root)

    archive_entry = workspace.archive / (
        name
        if name is not None
        else ('archive_' + datetime.now().isoformat('_', timespec='seconds'))
    )
    archive_entry.mkdir(exist_ok=False)

    for item in workspace_root.iterdir():
        if item.name == 'archive':
            continue

        if item.is_file():
            shutil.copy(item, archive_entry)

        elif item.is_dir():
            destination = archive_entry / item.name
            shutil.copytree(item, destination)

    click.echo(f'Successfully archived workspace "{workspace.name}"')
