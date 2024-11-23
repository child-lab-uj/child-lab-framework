from dataclasses import dataclass
from typing import Self

import numpy as np

from ....core import serialization
from ....core.transformation import EuclideanTransformation
from . import marker

# Design idea:
#   * `Configuration` is loaded from config file / constructed manually
#   * `AruDie` is constructed based on the configuration.
#     It stores static Euclidean transformations between planes containing its walls


@dataclass
class Tags:
    front: int
    back: int
    up: int
    down: int
    left: int
    right: int

    def __getitem__(self, wall_number: int) -> int:
        match wall_number:
            case 0:
                return self.front
            case 1:
                return self.back
            case 2:
                return self.up
            case 3:
                return self.down
            case 4:
                return self.left
            case 5:
                return self.right
            case _:
                raise KeyError()

    def serialize(self) -> dict[str, serialization.Value]:
        return {
            'front': self.front,
            'back': self.back,
            'up': self.up,
            'down': self.down,
            'left': self.left,
            'right': self.right,
        }

    @classmethod
    def deserialize(cls, data: dict[str, serialization.Value]) -> Self:
        match data:
            case {
                'front': int(front),
                'back': int(back),
                'up': int(up),
                'down': int(down),
                'left': int(left),
                'right': int(right),
            }:
                return cls(front, back, up, down, left, right)

            case other:
                raise serialization.DeserializeError(
                    f'Expected dictionary with front: int, back: int, up: int, down: int, left: int, right: int, got {other}'
                )


@dataclass(frozen=True)
class Configuration:
    size: float
    tags: Tags
    dictionary: marker.Dictionary

    def serialize(self) -> dict[str, serialization.Value]:
        return {
            'size': self.size,
            'tags': self.tags.serialize(),
            'dictionary': self.dictionary.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, serialization.Value]) -> Self:
        match data:
            case {
                'size': float(size),
                'tags': dict(tags),
                'dictionary': dict(dictionary),
                **_other,
            }:
                return cls(
                    size,
                    Tags.deserialize(tags),
                    marker.Dictionary.deserialize(dictionary),
                )

            case other:
                raise serialization.DeserializeError(
                    f'Expected dictionary with size: float, tags: serialized Tags and dictionary: serialized Dictionary, got {other}'
                )


class AruDie:
    configuration: Configuration

    static_transformations: dict[
        tuple[str, str],
        EuclideanTransformation,
    ]

    def __init__(self, configuration: Configuration) -> None:
        self.configuration = configuration
        self.static_transformations = _static_transformations(configuration)

    def serialize(self) -> dict[str, serialization.Value]:
        return {
            'configuration': self.configuration.serialize(),
            'static_transformations': [
                {
                    'from': from_name,
                    'to': to_name,
                    'transformation': transformation.serialize(),
                }
                for (
                    from_name,
                    to_name,
                ), transformation in self.static_transformations.items()
            ],
        }

    @classmethod
    def deserialize(cls, data: dict[str, serialization.Value]) -> Self:
        match data:
            case {
                'configuration': dict(serialized_configuration),
                'static_transformations': list(serialized_static_transformations),
                **_other,
            }:
                instance = cls(Configuration.deserialize(serialized_configuration))

                transformations = dict()

                for entry in serialized_static_transformations:
                    match entry:
                        case {
                            'from': str(from_name),
                            'to': str(to_name),
                            'transformation': dict(serialized_transformation),
                            **_other,
                        }:
                            transformations[from_name, to_name] = (
                                EuclideanTransformation.deserialize(
                                    serialized_transformation
                                )
                            )

                        case other:
                            raise serialization.DeserializeError(
                                f'Invalid transformation encoding encountered: {other}'
                            )

                instance.static_transformations = transformations
                return instance

            case other:
                raise serialization.DeserializeError(
                    'Expected dictionary with configuration: serialized Configuration '
                    f'and static_transformations: list[serialized EuclideanTransformation], got {other}'
                )


def _static_transformations(
    configuration: Configuration,
) -> dict[tuple[str, str], EuclideanTransformation]:
    # https://dugas.ch/transform_viewer/multi.html
    # Returns transforms from arudie face 0 to subsequent faces i.e. list[1] -> to face 1

    full = configuration.size
    half = full / 2.0

    from_zero = [
        # 0 -> identity
        EuclideanTransformation(
            np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        ),
        # 1
        EuclideanTransformation(
            np.array(
                [
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, -1, 0],
                ],
                dtype=np.float32,
            ),
            np.array([0, half, -half], dtype=np.float32),
        ),
        # 2
        EuclideanTransformation(
            np.array(
                [
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1],
                ],
                dtype=np.float32,
            ),
            np.array([0, 0, -full], dtype=np.float32),
        ),
        # 3
        EuclideanTransformation(
            np.array(
                [
                    [1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0],
                ],
                dtype=np.float32,
            ),
            np.array([0, -half, -half], dtype=np.float32),
        ),
        # 4
        EuclideanTransformation(
            np.array(
                [
                    [0, 0, 1],
                    [0, 1, 0],
                    [-1, 0, 0],
                ],
                dtype=np.float32,
            ),
            np.array([half, 0, -half], dtype=np.float32),
        ),
        # 5
        EuclideanTransformation(
            np.array(
                [
                    [0, 0, -1],
                    [0, 1, 0],
                    [1, 0, 0],
                ],
                dtype=np.float32,
            ),
            np.array([-half, 0, -half], dtype=np.float32),
        ),
    ]

    tags = configuration.tags
    wall_names = ['marker' + str(tags[i]) for i in range(6)]

    result = {}

    result.update(
        {
            (wall_names[i], wall_names[i]): EuclideanTransformation(
                np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
            )
            for i in range(6)
        }
    )

    from_zeros_without_zero = enumerate(from_zero)
    next(from_zeros_without_zero)  # awful imperative designs ;v

    result.update(
        {
            (wall_names[0], wall_names[i]): transformation
            for i, transformation in from_zeros_without_zero
        }
    )

    for i in range(1, 6):
        i_name = wall_names[i]
        i_to_zero = from_zero[i].inverse

        for j in range(i + 1, 6):
            j_name = wall_names[j]
            zero_to_j = from_zero[j]

            result[i_name, j_name] = zero_to_j @ i_to_zero

    return result
