import click

from .commands import (
    calibrate_camera,
    estimate_transformations,
    generate_pointcloud,
    process,
    tune_calibration,
    video,
    visualize,
    workspace,
)
from .log import setup_logging


@click.group('child-lab')
def cli() -> None: ...


cli.add_command(calibrate_camera)
cli.add_command(estimate_transformations)
cli.add_command(generate_pointcloud)
cli.add_command(process)
cli.add_command(tune_calibration)
cli.add_command(video)
cli.add_command(visualize)
cli.add_command(workspace)


def child_lab() -> None:
    setup_logging()
    cli(max_content_width=120)
