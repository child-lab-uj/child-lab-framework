from collections.abc import Callable, Hashable, Iterable
from copy import deepcopy
from functools import reduce
from typing import Self

from more_itertools import pairwise
from rustworkx import PyDiGraph, digraph_dijkstra_shortest_paths, is_connected

from transformation_buffer.rigid_model import RigidModel
from transformation_buffer.transformation import Transformation

# TODO: Implement `pyserde` serialization.
# TODO: Implement path compression and verification.


class Buffer[T: Hashable]:
    __frames_of_reference: set[T]
    __frame_names_to_node_indices: dict[T, int]
    __connections: PyDiGraph[T, Transformation]

    def __init__(self, frames_of_reference: Iterable[T] | None = None) -> None:
        nodes = (
            set() if frames_of_reference is None else deepcopy(set(frames_of_reference))
        )
        n_nodes = len(nodes)

        connections = PyDiGraph[T, Transformation](
            multigraph=False,  # Think about it later
            node_count_hint=n_nodes,
            edge_count_hint=n_nodes,
        )

        node_indices = connections.add_nodes_from(nodes)

        self.__frames_of_reference = nodes
        self.__frame_names_to_node_indices = dict(zip(nodes, node_indices))
        self.__connections = connections

    @property
    def connected(self) -> bool:
        return is_connected(self.__connections.to_undirected(multigraph=False))

    @property
    def frames_of_reference(self) -> frozenset[T]:
        # Construction of frozenset from set of strings is *somehow* optimized.
        # Perhaps there's no need for caching this property.
        return frozenset(self.__frames_of_reference)

    def add_object(self, model: RigidModel[T]) -> Self:
        for from_to, transformation in model.transformations().items():
            self[from_to] = transformation

        return self

    def add_frame_of_reference(self, frame: T) -> Self:
        self.__add_frame_of_reference(frame)
        return self

    def add_transformation(
        self,
        from_frame: T,
        to_frame: T,
        transformation: Transformation,
    ) -> Self:
        return self

    def __add_frame_of_reference(self, frame: T) -> int:
        if (index := self.__frame_index(frame)) is not None:
            return index

        self.__frames_of_reference.add(frame)
        index = self.__connections.add_node(frame)
        self.__frame_names_to_node_indices[frame] = index

        return index

    def __frame_index(self, frame: T) -> int | None:
        return self.__frame_names_to_node_indices.get(frame, None)

    def __setitem__(self, from_to: tuple[T, T], transformation: Transformation) -> None:
        from_id, to_id = from_to

        from_index = self.__frame_index(from_id) or self.__add_frame_of_reference(from_id)
        to_index = self.__frame_index(to_id) or self.__add_frame_of_reference(to_id)

        self.__connections.add_edge(from_index, to_index, transformation)
        self.__connections.add_edge(to_index, from_index, transformation.inverse())

    def __getitem__(self, from_to: tuple[T, T]) -> Transformation | None:
        from_id, to_id = from_to

        from_index = self.__frame_index(from_id)
        to_index = self.__frame_index(to_id)

        if from_index is None or to_index is None:
            return None

        if from_index == to_index:
            return Transformation.identity().clone()

        connections = self.__connections

        if self.__connections.has_edge(from_index, to_index):
            return self.__connections.get_edge_data(from_index, to_index)

        path_mapping = digraph_dijkstra_shortest_paths(
            connections,
            from_index,
            to_index,
        )

        if to_index not in path_mapping:
            return None

        shortest_path = path_mapping[to_index]

        return map_reduce(  # type: ignore[no-any-return]  # MyPy complains about the possible `Any`.
            lambda nodes: connections.get_edge_data(*nodes),
            lambda t1, t2: t1 @ t2,
            pairwise(shortest_path),
        )

    def __contains__(self, from_to: tuple[T, T]) -> bool:
        from_id, to_id = from_to

        from_index = self.__frame_index(from_id)
        to_index = self.__frame_index(to_id)

        if from_index is None or to_index is None:
            return False

        return self.__connections.has_edge(from_index, to_index)


def map_reduce[U, V](
    map_function: Callable[[U], V],
    reduce_function: Callable[[V, V], V],
    iterable: Iterable[U],
) -> V:
    return reduce(reduce_function, map(map_function, iterable))
