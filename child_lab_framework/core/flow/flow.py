from collections.abc import Iterable, Mapping
import time
import sys
from typing import Any, Callable

from ...typing.flow import Component
from ...typing.stream import Fiber
from .compilation import compiled_flow


class Machinery:
    components: dict[str, Component]
    streams: dict[str, Fiber]
    inputs: list[str]
    outputs: list[str]
    dependencies: dict[str, tuple[str, ...]]
    flow_controller: Fiber[None, bool]

    def __init__(
        self,
        components: Iterable[tuple[str, Component]],
        inputs: Iterable[str],
        outputs: Iterable[str],
        dependencies: Mapping[str, tuple[str, ...]]
    ) -> None:
        self.components = dict(components)
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.dependencies = dict(dependencies)
        # TODO: graph computations: DAG check, optimization? etc.

        self.streams = self.__open_streams()
        self.flow_controller = self.__compile()(self.components)

    def __open_streams(self) -> dict[str, Fiber]:
        return {
            name: component.stream()
            for name, component in self.components.items()
        }

    def __compile(self) -> Callable[[dict[str, Component]], Fiber[None, bool]]:
        exec(compiled_flow(
            self.components.keys(),
            self.inputs,
            self.outputs,
            self.dependencies,
        ))

        return locals()['__step']

    def run(self) -> None:
        controller = self.flow_controller
        controller.send(None)

        step = 0
        start = time.time()

        while True:
            status = controller.send(None)

            step += 1
            elapsed = time.time() - start
            sys.stdout.write(f'\rstep {step} (elapsed time: {elapsed:.2f} s)')

            if status:
                break
