import pytest
import serde
from rustworkx import PyDiGraph
from serde.json import from_json, to_json
from serde.yaml import to_yaml
from syrupy.assertion import SnapshotAssertion
from transformation_buffer.serialization.graph import deserialize_graph, serialize_graph


# A wrapper class to test field serialization.
# `PyDiGraph` cannot be serialized to any concrete format itself.
@serde.serde
class WrapperClass[N, E]:
    graph: PyDiGraph[N, E] = serde.field(
        serializer=serialize_graph,
        deserializer=deserialize_graph,
    )


@pytest.fixture
def empty_graph() -> PyDiGraph[int, int]:
    return PyDiGraph[int, int]()


@pytest.fixture
def simple_graph() -> PyDiGraph[str, str]:
    graph = PyDiGraph[str, str]()

    a = graph.add_node('a')
    b = graph.add_node('b')
    c = graph.add_node('c')
    d = graph.add_node('d')
    e = graph.add_node('e')

    graph.add_edge(a, b, 'edge1')
    graph.add_edge(a, c, 'edge2')
    graph.add_edge(d, e, 'edge3')

    return graph


def test_serialize_with_empty_graph_json(
    empty_graph: PyDiGraph[int, int],
    snapshot: SnapshotAssertion,
) -> None:
    obj = WrapperClass(empty_graph)
    serialized_obj = to_json(obj)
    assert serialized_obj == snapshot


def test_serialize_with_simple_graph_json(
    simple_graph: PyDiGraph[str, str],
    snapshot: SnapshotAssertion,
) -> None:
    obj = WrapperClass(simple_graph)
    serialized_obj = to_json(obj)
    assert serialized_obj == snapshot


def test_deserialize_with_empty_graph_json() -> None:
    data = """
        {
            "graph": {
                "check_cycle": false,
                "multigraph": true,
                "nodes": [],
                "edges": []
            }
        }
    """

    graph = from_json(WrapperClass[int, int], data).graph

    assert not graph.check_cycle
    assert graph.multigraph
    assert graph.nodes() == []
    assert graph.edges() == []


def test_deserialize_with_simple_graph_json() -> None:
    data = """
        {
            "graph": {
                "check_cycle": false,
                "multigraph": true,
                "nodes": [
                    [0, "a"],
                    [1, "b"],
                    [2, "c"],
                    [3, "d"],
                    [4, "e"]
                ],
                "edges": [
                    [[0, 1], "edge1"],
                    [[0, 2], "edge2"],
                    [[3, 4], "edge3"]
                ]
            }
        }
    """

    graph = from_json(WrapperClass[str, str], data).graph

    assert not graph.check_cycle
    assert graph.multigraph
    assert graph.nodes() == ['a', 'b', 'c', 'd', 'e']
    assert graph.edges() == ['edge1', 'edge2', 'edge3']


def test_serialize_with_empty_graph_yaml(
    empty_graph: PyDiGraph[int, int],
    snapshot: SnapshotAssertion,
) -> None:
    obj = WrapperClass(empty_graph)
    serialized_obj = to_yaml(obj)
    assert serialized_obj == snapshot
