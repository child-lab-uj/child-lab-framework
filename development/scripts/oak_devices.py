#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = ["rich", "numpy", "depthai>=3.0.0rc4"]
#
# [[tool.uv.index]]
# name = "luxonis-depthai"
# url = "https://artifacts.luxonis.com/artifactory/luxonis-python-release-local"
# explicit = false
# ///


import depthai as dai
from rich import print


def main() -> None:
    devices = dai.Device.getAllAvailableDevices()

    if len(devices) == 0:
        print('[bold red]No devices available[/bold red]')
        return

    print('[bold]Available devices:[/bold]')

    for i, device in enumerate(devices):
        print(
            f'Device {i + 1}: [blue]{device.deviceId}[/blue] @ [green]{device.name}[/green]'
        )


if __name__ == '__main__':
    main()
