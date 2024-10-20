import time
from collections.abc import Callable, Iterable

import networkx as nx

from ...logging.logger import Logger
from ...typing.flow import Component
from ...typing.stream import Fiber
from .compilation import compiled_flow

type ComponentDefinition = (
    tuple[str, Component]
    | tuple[str, Component, str]
    | tuple[str, Component, tuple[str, ...]]
)


class Machinery:
    components: dict[str, Component]
    dependencies: nx.DiGraph
    flow: nx.DiGraph
    flow_controller: Fiber[None, bool]

    def __init__(
        self,
        definitions: list[ComponentDefinition],
    ) -> None:
        self.dependencies = self.__build_dependencies(definitions)
        self.flow = self.dependencies.reverse()
        self.components = dict((name, component) for name, component, *_ in definitions)
        self.flow_controller = self.__compile()(self.components)

    def __build_dependencies(self, definitions: list[ComponentDefinition]) -> nx.DiGraph:
        dependencies: dict[str, Iterable[str]] = dict()

        for definition in definitions:
            match definition:
                case name, _, str(dependency):
                    dependencies[name] = (dependency,)

                case name, _, tuple(deps):
                    dependencies[name] = deps

        graph = nx.DiGraph(dependencies)
        assert nx.is_directed_acyclic_graph(graph)  # TODO: Throw a proper exception

        return graph

    def __compile(self) -> Callable[[dict[str, Component]], Fiber[None, bool]]:
        exec(compiled_flow(self.flow, self.dependencies, function_name='__step'))
        return locals()['__step']

    async def run(self) -> None:
        controller = self.flow_controller
        await controller.asend(None)

        step = 0
        start = time.time()
        end: float

        while True:
            status = await controller.asend(None)

            end = time.time()
            elapsed = end - start

            Logger.info(f'Step {step}, elapsed time: {elapsed:.2f} s')

            if status:
                break

            step += 1
            start = end
