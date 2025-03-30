from pathlib import Path

import click


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
    pass
