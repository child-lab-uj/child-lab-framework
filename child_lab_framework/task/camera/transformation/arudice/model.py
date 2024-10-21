from dataclasses import dataclass

from ...detection import marker
from .. import transformation

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


# TODO: `Configuration` loading
@dataclass(frozen=True)
class Configuration:
    size: float
    tags: Tags
    dictionary: marker.Dictionary


class AruDie:
    configuration: Configuration

    static_transformations: dict[
        tuple[int, int],
        transformation.Result,
    ]

    # TODO: implement static transformations calculation
    def __init__(self, configuration: Configuration) -> None: ...

    # TODO: think about implementing `Transformation` protocol on `AruDie`
    # to be able to pass it to function expecting transformation and write `arudie.project(...)` etc.
