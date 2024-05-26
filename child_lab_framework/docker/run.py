from typing import Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from docker.models.containers import Container

from child_lab_framework.docker import build
from child_lab_framework.docker.client import get_default_client
from child_lab_framework.logging import Logger


def kill_callback(container: "Container", widget_name: str, address: str, port: int):
    container.kill()
    Logger.info(f"Killed container with {widget_name} running at {address}:{port}.")


# Build and run a dockerized widget, return a callback to kill the widget
def run(
    widget_name: str,
    port: int,
    address: str = "127.0.0.1",
    # command: Optional[str] = None,
    **env,
) -> Callable[[None], None]:

    tag = build(widget_name)
    client = get_default_client()

    container = client.containers.run(
        tag,
        network_mode="host",
        environment={"CLF_ADDRESS": address, "CLF_PORT": str(port), **env},
        # publish_all_ports=True,
        detach=True,
    )
    Logger.info(f"Container with {widget_name} running at {address}:{port}...")

    return lambda: kill_callback(container, widget_name, address, port)


if __name__ == "__main__":
    from time import sleep

    handle = run("test-widget", 15101)
    sleep(2)
    handle()
