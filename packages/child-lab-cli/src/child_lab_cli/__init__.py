import click

from .commands import (
    calibrate_camera,
    estimate_transformations,
    generate_pointcloud,
    process,
    video,
    visualize,
    workspace,
)


@click.group('child-lab')
def cli() -> None: ...


cli.add_command(calibrate_camera)
cli.add_command(estimate_transformations)
cli.add_command(generate_pointcloud)
cli.add_command(process)
cli.add_command(video)
cli.add_command(visualize)
cli.add_command(workspace)


def child_lab() -> None:
    cli(max_content_width=120)
