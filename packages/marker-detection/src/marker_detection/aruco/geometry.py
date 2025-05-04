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

    def rotation_quaternion(self) -> Float[numpy.ndarray, '4']:
        # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion

        rotation = self.rotation

        t = rotation[0, 0] + rotation[1, 1] + rotation[2, 2]

        if t > 0.0:
            r = numpy.sqrt(1.0 + t)
            s = 1 / (2 * r)
            w = r / 2
            x = (rotation[2, 1] - rotation[1, 2]) * s
            y = (rotation[0, 2] - rotation[2, 0]) * s
            z = (rotation[1, 0] - rotation[0, 1]) * s
        else:
            r = numpy.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            s = 1 / (2 * r)
            w = (rotation[2, 1] - rotation[1, 2]) * s
            x = r / 2
            y = (rotation[0, 1] + rotation[1, 0]) * s
            z = (rotation[0, 2] + rotation[2, 0]) * s

        result = numpy.stack((w, x, y, z))

        return result


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
