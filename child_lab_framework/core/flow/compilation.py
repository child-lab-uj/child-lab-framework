from collections.abc import Iterable, Mapping
from types import CodeType
from typing import Callable, TypeVar
from textwrap import dedent

from ...typing.stream import Fiber
from ...typing.flow import Component


def stream_identifier_bindings(components: Iterable[str], tabs: int) -> str:
    indent = '\t' * tabs

    return '\n'.join(
        f'{indent}{name}_fiber = components[\'{name}\'].stream()\n' +
        f'{indent}await {name}_fiber.asend(None)'
        for name in components
    )


def checked_inputs(inputs: list[str], tabs: int) -> list[str]:
    indent = '\t' * tabs

    return [
        f'{indent}({name}_value := await {name}_fiber.asend(None)) is not None'
        for name in inputs
    ]


def main_loop_header(inputs: list[str], tabs: int) -> str:
    indent = '\t' * tabs
    input_lines = checked_inputs(inputs, tabs + 2)

    return (
        f'{indent}while (\n' +
        ' and\n'.join(input_lines) +
        '\n' +
        f'{indent}):'
    )


def topological_order_from_root(
    inputs: set[str],
    dependencies: dict[str, tuple[str, ...]],
    root: str,
    visited: dict[str, bool]
) -> list[str]:
    if (
        visited.get(root) is True or
        root in inputs
    ):
        return []

    if root not in dependencies:
        # TODO: warn about dead component
        return []

    visited[root] = True

    return sum(
        (
            topological_order_from_root(
                inputs,
                dependencies,
                dependency,
                visited
            )
            for dependency
            in dependencies[root]
        ),
        start=[]
    ) + [root]


def ordered_components(inputs: list[str], outputs: list[str], dependencies: dict[str, tuple[str, ...]]) -> list[str]:
    visited = dict()

    return sum(
        (
            topological_order_from_root(
                set(inputs),
                dependencies,
                root,
                visited
            )
            for root
            in outputs
        ),
        start=[]
    )


def packet_of_dependencies_to_send(component: str, dependencies: tuple[str, ...]) -> str:
    return (
        '(' + ', '.join(
            f'{dependency}_value'
            for dependency in dependencies
        ) + ')'
    )


def main_loop_body(inputs: list[str], outputs: list[str], dependencies: dict[str, tuple[str, ...]], tabs: int) -> str:
    indent = '\t' * tabs

    sends = '\n'.join(
        f'{indent}{name}_value = await {name}_fiber.asend({packet_of_dependencies_to_send(name, dependencies[name])})'
        for name in ordered_components(
            inputs,
            outputs,
            dependencies
        )
    )

    return sends + '\n' + f'{indent}yield False'


def compiled_flow(
    components: Iterable[str],
    inputs: list[str],
    outputs: list[str],
    dependencies: dict[str, tuple[str, ...]],
    *,
    function_name: str = '__step'
) -> CodeType:
    header = f'async def {function_name}(components: dict[str, Component]) -> Fiber[None, bool]:'
    ending = f'\twhile True: yield True'

    source_code = '\n'.join((
        header,
        stream_identifier_bindings(components, 1),
        main_loop_header(inputs, 1),
        main_loop_body(inputs, outputs, dependencies, 2),
        ending
    ))

    print('[INFO] Flow source code:', source_code, sep='\n')

    return compile(
        source_code,
        filename='compiled_flow',
        mode='single',
        optimize=-1
    )
