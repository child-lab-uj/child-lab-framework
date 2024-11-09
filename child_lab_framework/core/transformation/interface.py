from typing import Protocol, Self

from ...typing.array import FloatArray2
from ..calibration import Calibration
from .transformation import Transformation


class Transformable(Protocol):
    def transform(self, transformation: Transformation) -> Self: ...


class Projectable[T: 'Unprojectable'](Protocol):
    @property
    def flat_points(self) -> FloatArray2: ...
    def project(self, calibration: Calibration) -> T: ...


class Unprojectable[T](Protocol):
    def unproject(self, calibration: Calibration, depth: FloatArray2) -> T: ...


# A workaround - Python doesn't have a notion of type intersection.
# Cannot just write `T: Projectable + Transformable`.
# Remove this protocol as soon as it's possible. Hopefully...
class ProjectableAndTransformable(Projectable, Transformable, Protocol): ...
