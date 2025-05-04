from dataclasses import dataclass

import numpy
import scipy.spatial.transform as scipy_transform
from jaxtyping import Float

type IntrinsicsMatrix = Float[numpy.ndarray, '3 3']
type DistortionCoefficients = Float[numpy.ndarray, '5']


@dataclass(slots=True)
class Transformation:
    rotation: Float[numpy.ndarray, '3 3']
    translation: Float[numpy.ndarray, '3']

    def euler_angles(self) -> Float[numpy.ndarray, '3']:
        return (  # type: ignore[no-any-return]
            scipy_transform.Rotation.from_matrix(self.rotation).as_euler(
                'xyz',
                degrees=False,
            )
        )


class MarkerRigidModel:
    square_size: float
    depth: float
    border: float
    coordinates: Float[numpy.ndarray, '3 4']

    def __init__(self, square_size: float, depth: float, border: float) -> None:
        self.square_size = square_size
        self.depth = depth
        self.border = border

        self.coordinates = numpy.array(
            [
                [-square_size / 2.0, square_size / 2.0, depth],
                [square_size / 2.0, square_size / 2.0, depth],
                [square_size / 2.0, -square_size / 2.0, depth],
                [-square_size / 2.0, -square_size / 2.0, depth],
            ],
            dtype=numpy.float32,
        )
