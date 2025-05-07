from pathlib import Path

import click

from child_lab_cli.workspace.model import Workspace


@click.command('process')
@click.argument('workspace-root', type=Path, metavar='<workspace>')
@click.argument('video_names', type=str, nargs=-1, metavar='<videos>')
@click.option(
    '--device',
    type=str,
    default='cpu',
    help='Torch device to use for tensor computations',
)
@click.option(
    '--skip',
    type=int,
    required=False,
    help='Seconds of videos to skip at the beginning',
)
def process(
    workspace_root: Path,
    video_names: list[str],
    device: str,
) -> None:
    workspace = Workspace.in_directory(workspace_root)
