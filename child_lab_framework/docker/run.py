from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from docker.models.containers import Container

from ..docker.build import build
from ..logging import Logger
from .client import get_default_client


def kill_callback(container: 'Container', widget_name: str, address: str, port: int):
    container.kill()
    Logger.info(f'Killed container with {widget_name} running at {address}:{port}.')


# Build and run a dockerized widget, return a callback to kill the widget
def run(
    widget_name: str,
    port: int,
    address: str = '127.0.0.1',
    # command: Optional[str] = None,
    **env,
) -> Callable[[], None]:
    tag = build(widget_name)
    client = get_default_client()

    container = client.containers.run(
        tag,
        network_mode='host',
        environment={'CLF_ADDRESS': address, 'CLF_PORT': str(port), **env},
        # publish_all_ports=True,
        detach=True,
    )
    Logger.info(f'Container with {widget_name} running at {address}:{port}...')

    return lambda: kill_callback(container, widget_name, address, port)


def main() -> None:
    from time import sleep

    handle = run('test-widget', 15101)
    sleep(2)
    handle()


if __name__ == '__main__':
    main()
