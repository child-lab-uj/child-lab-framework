from functools import reduce
from typing import Hashable, Self

import networkx as nx
from more_itertools import pairwise
from more_itertools.more import filter_map

from .. import serialization
from . import Transformation
from .error import reprojection_error as _reprojection_error
from .interface import ProjectableAndTransformable


# T appears both as a method argument and return type, therefore it must be invariant.
class Buffer[T: Hashable]:
    __frames_of_reference: set[T]
    __connections: 'nx.DiGraph[T]'

    __compress_paths: bool
    __strict_check: bool

    def __init__(
        self,
        frames_of_reference: set[T] | None = None,
        *,
        compress_paths: bool = True,
        strict_check: bool = False,
    ) -> None:
        self.__frames_of_reference = frames_of_reference or set()
        self.__connections = nx.DiGraph()
        self.__connections.add_nodes_from(self.__frames_of_reference)

        self.__compress_paths = compress_paths
        self.__strict_check = strict_check

    @property
    def connected(self) -> bool:
        return nx.is_connected(self.__connections.to_undirected(as_view=True))

    @property
    def frames_of_reference(self) -> frozenset[T]:
        # Construction of frozenset from set of strings is *somehow* optimized.
        # Perhaps there's no sense in caching this property.
        return frozenset(self.__frames_of_reference)

    @property
    def connections(self) -> 'nx.DiGraph[T]':
        return nx.restricted_view(self.__connections, [], [])

    def add_frame_of_reference(self, name: T) -> Self:
        self.__frames_of_reference.add(name)
        self.__connections.add_node(name)
        return self

    def __contains__(self, from_to: tuple[T, T]) -> bool:
        return nx.has_path(self.__connections, *from_to)

    def __setitem__(
        self,
        from_to: tuple[T, T],
        transformation: Transformation,
    ) -> None:
        from_node, to_node = from_to

        self.__frames_of_reference.add(from_node)
        self.__frames_of_reference.add(to_node)
        self.__connections.add_edge(from_node, to_node, transformation=transformation)

        if not self.__contains__((to_node, from_node)):
            self.__connections.add_edge(
                to_node,
                from_node,
                transformation=transformation.inverse,
            )

    def __getitem__(self, from_to: tuple[T, T]) -> Transformation | None:
        connections = self.__connections

        match connections.get_edge_data(*from_to):
            case {'transformation': transformation} if isinstance(
                transformation, Transformation
            ):
                maybe_result = transformation

            case _:
                maybe_result = None

        if maybe_result is not None:
            return maybe_result

        if not nx.has_path(connections, *from_to):
            return None

        transformation = reduce(
            lambda x, y: y @ x,  # operator.matmul is not properly typed
            filter_map(
                self.__getitem__,
                pairwise(nx.shortest_path(connections, *from_to, weight=None)),
            ),
        )

        if self.__compress_paths:
            self.__connections.add_edge(*from_to, transformation=transformation)

        self.__frames_of_reference.add(from_to[0])
        self.__frames_of_reference.add(from_to[1])

        return transformation

    def reprojection_error[U: ProjectableAndTransformable](
        self,
        evaluated_frame: T,
        referential_frame: T,
        evaluated_object: U,
        referential_object: U,
    ) -> float:
        transformation = self.__getitem__((evaluated_frame, referential_frame))
        if transformation is None:
            return float('inf')

        return _reprojection_error(evaluated_object, referential_object, transformation)

    def update_transformation_if_better[U: ProjectableAndTransformable](
        self,
        from_frame: T,
        to_frame: T,
        from_object: U,
        to_object: U,
        transformation: Transformation,
    ) -> None:
        new_error = _reprojection_error(from_object, to_object, transformation)
        current_error = self.reprojection_error(
            from_frame,
            to_frame,
            from_object,
            to_object,
        )

        if new_error < current_error:
            self.__setitem__((from_frame, to_frame), transformation)

        inverse_transformation = transformation.inverse

        new_error = _reprojection_error(to_object, from_object, inverse_transformation)
        current_error = self.reprojection_error(
            to_frame,
            from_frame,
            to_object,
            from_object,
        )

        if new_error < current_error:
            self.__setitem__((to_frame, from_frame), inverse_transformation)

    def serialize(self) -> dict[str, serialization.Value]:
        return {
            'frames_of_reference': list(map(str, self.__frames_of_reference)),
            'transformations': {
                ' -> '.join(key): value['transformation'].serialize()
                for key, value in dict(self.__connections.edges).items()
            },
            'compress_paths': self.__compress_paths,
            'strict_check': self.__strict_check,
        }

    # Deserialization of a parametrized class cannot be done well in Python.
    # Guarantees:
    # 1. all __frames_of_reference are the same type
    # 2. __frames_of_reference is consistent with __connections
    #
    # Caller receives Buffer[Hashable] and has to downcast it manually to a proper type.
    @classmethod
    def deserialize(cls, data: dict[str, serialization.Value]) -> Self:
        match data:
            # `frames_of_reference` is not secessarily a `Value` here - it must have been *read* as a Value,
            # but might then be transformed into e.g. an enum variant.

            case {
                'frames_of_reference': list(encoded_frames_of_reference),
                'transformations': dict(encoded_transformations),
                'compress_paths': bool(compress_paths),
                'strict_check': bool(strict_check),
                **_other,
            }:
                frame_types = tuple(set(map(type, encoded_frames_of_reference)))

                match frame_types:
                    case (frame_type,):
                        if not issubclass(frame_type, Hashable):
                            raise serialization.DeserializeError(
                                f'Expected hashable frame of reference identifier type, got {frame_type}'
                            )

                    case other:
                        raise serialization.DeserializeError(
                            f'Expected all frames of reference identifiers to be the same type, got types: {other}'
                        )

                frames_of_reference: set[Hashable] = set(encoded_frames_of_reference)  # type: ignore
                transformations: dict[tuple[Hashable, Hashable], Transformation] = {}

                for key, value in encoded_transformations.items():
                    from_node, to_node = key.split(' -> ')

                    if from_node not in frames_of_reference:
                        raise serialization.DeserializeError(
                            f'Unknown frame of reference "{from_node}" found'
                        )

                    if to_node not in frames_of_reference:
                        raise serialization.DeserializeError(
                            f'Unknown frame of reference "{to_node}" found'
                        )

                    if not isinstance(value, dict):
                        raise serialization.DeserializeError(
                            f'Expected transformation encoded as dict[str, Value], got {type[value]}'
                        )

                    maybe_transformation = Transformation.deserialize(value)
                    if maybe_transformation is not None:
                        transformations[(from_node, to_node)] = maybe_transformation

                instance = Buffer(
                    frames_of_reference,
                    compress_paths=compress_paths,
                    strict_check=strict_check,
                )

                connections = nx.DiGraph()

                for from_to, transformation in transformations.items():
                    connections.add_edge(*from_to, transformation=transformation)

                instance.__connections = connections

                return instance  # type: ignore

            case other:
                raise serialization.DeserializeError(
                    'Expected dictionary with: '
                    'frames_of_reference: list[T] where T: Hashable, '
                    'transformations: dict[str, dict[str, Value]], '
                    'compress_paths: bool '
                    'and strict_check: bool '
                    'keys, got {}',
                    other,
                )
