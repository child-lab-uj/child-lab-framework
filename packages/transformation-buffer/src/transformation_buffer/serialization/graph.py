from typing import Any

import serde
from plum import dispatch
from rustworkx import PyDiGraph

from transformation_buffer.transformation import Transformation


def init() -> None:
    serde.add_serializer(Serializer())
    serde.add_deserializer(Deserializer())


class Serializer:
    @dispatch
    def serialize(self, value: PyDiGraph) -> dict[str, Any]:
        return serialize_graph(value)


class Deserializer:
    @dispatch
    def deserialize(self, cls: type[PyDiGraph], value: Any) -> PyDiGraph:
        return deserialize_graph(value)


def serialize_graph(graph: PyDiGraph) -> dict[str, Any]:
    nodes = [(index, graph.get_node_data(index)) for index in graph.node_indices()]
    edges = [
        (from_to, serde.to_dict(graph.get_edge_data(*from_to)))
        for from_to in graph.edge_list()
    ]

    return {
        'check_cycle': graph.check_cycle,
        'multigraph': graph.multigraph,
        'nodes': nodes,
        'edges': edges,
    }


# INVARIANT: Properly deserializes only graphs with Transformations as edge poayloads.
def deserialize_graph(value: Any) -> PyDiGraph:
    match value:
        case {
            'check_cycle': bool(check_cycle),
            'multigraph': bool(multigraph),
            'nodes': list(nodes),
            'edges': list(edges),
        }:
            graph = PyDiGraph(
                check_cycle,
                multigraph,
                node_count_hint=len(nodes),
                edge_count_hint=len(edges),
            )

            old_node_indices_to_new = {}

            for old_index, node in sorted(nodes, key=lambda entry: entry[0]):
                new_index = graph.add_node(node)
                old_node_indices_to_new[old_index] = new_index

            for (old_a, old_b), edge in edges:
                new_a = old_node_indices_to_new[old_a]
                new_b = old_node_indices_to_new[old_b]

                # TODO: Unhack
                graph.add_edge(new_a, new_b, serde.from_dict(Transformation, edge))

            return graph

        case _:
            raise serde.SerdeError('')
