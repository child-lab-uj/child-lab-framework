import subprocess
from os import listdir

from ..logging import Logger
from ..util import WIDGETS_DIR, CLFException
from .client import get_default_client


# Builds a docker widget if not already built, returns its tag.
def build(widget_name: str) -> str:
    try:
        widgets = listdir(WIDGETS_DIR)
    except FileNotFoundError as e:
        raise CLFException(f'WIDGETS_DIR does not exist at {WIDGETS_DIR}.') from e

    if widget_name not in widgets:
        raise CLFException(f'Widget `{widget_name}` not found in `{WIDGETS_DIR}`.')

    widget_path = WIDGETS_DIR / widget_name

    rev = (
        subprocess.check_output('git rev-parse HEAD', shell=True, cwd=widget_path)
        .strip()
        .decode('utf-8')
    )

    docker_client = get_default_client()

    expected_tag = f'{widget_name}:{rev}'

    image_tags: list[str] = list(  # pyright: ignore
        map(
            lambda image: image.tags[0] if image.tags else None,
            docker_client.images.list(),
        )
    )

    if expected_tag in image_tags:
        Logger.info(f'Found image {expected_tag}, omitting the build step.')
        return expected_tag

    Logger.info(f'Image {expected_tag} not found, building, it may take some time.')
    docker_client.images.build(
        path=str(widget_path),
        tag=expected_tag,
    )

    Logger.info(f'Image {expected_tag} built successfully.')
    return expected_tag


if __name__ == '__main__':
    build('test-widget')
