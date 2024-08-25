from collections.abc import Iterable, Generator
from types import CodeType
from typing import Callable, TypeVar
from textwrap import dedent
import networkx as nx

from ...typing.stream import Fiber
from ...typing.flow import Component


def stream_identifier_bindings(components: Iterable[str], tabs: int) -> str:
    indent = '\t' * tabs

    return '\n'.join(
        f'{indent}{name}_fiber = components[\'{name}\'].stream()\n' +
        f'{indent}await {name}_fiber.asend(None)'
        for name in components
    )


def main_loop_header(flow: nx.DiGraph, tabs: int) -> str:
    indent = '\t' * tabs

    checked_inputs = (
        f'{indent}\t\t({name}_value := await {name}_fiber.asend(None)) is not None'
        for name in flow.nodes
        if flow.in_degree(name) == 0
    )

    return (
        f'{indent}while (\n' +
        ' and\n'.join(checked_inputs) +
        '\n' +
        f'{indent}):'
    )


def packet_of_dependencies_to_send(component: str, dependencies: Iterable[str]) -> str:
    return (
        '(' + ', '.join(
            f'{name}_value'
            for name in dependencies
        ) + ')'
    )


def main_loop_body(flow: nx.DiGraph, dependencies: nx.DiGraph, tabs: int) -> str:
    indent = '\t' * tabs

    dependency_layers: list[list[str]] = []
    independent_nodes = []

    for node in nx.topological_sort(flow):
        if flow.in_degree(node) == 0:
            continue

        if any(flow.has_edge(other, node) for other in independent_nodes):
            dependency_layers.append(independent_nodes)
            independent_nodes = []

        independent_nodes.append(node)

    dependency_layers.append(independent_nodes)

    gathers: list[str] = []

    for layer in dependency_layers:
        if len(layer) == 0:
            continue

        value_identifiers = ', '.join(f'{name}_value' for name in layer)

        sends = ', '.join(
            f'{name}_fiber.asend({packet_of_dependencies_to_send(name, dependencies[name])})'
            for name in layer
        )

        gathers.append(f'{indent}({value_identifiers},) = await asyncio.gather({sends})')

    return '\n'.join(gathers) + '\n' + f'{indent}yield False'


def compiled_flow(
    flow: nx.DiGraph,
    dependencies: nx.DiGraph,
    *,
    function_name: str = '__step'
) -> CodeType:
    header = f'async def {function_name}(components: dict[str, Component]) -> Fiber[None, bool]:'
    imports = '\timport asyncio'
    ending = f'\twhile True: yield True'

    inputs = (node for node in flow.nodes if flow.in_degree(node) == 0)

    source_code = '\n'.join((
        header,
        imports,
        stream_identifier_bindings(flow.nodes, 1),
        main_loop_header(flow, 1),
        main_loop_body(flow, dependencies, 2),
        ending
    ))

    return compile(
        source_code,
        filename='compiled_flow',
        mode='single',
        optimize=-1
    )
