from dataclasses import dataclass
from typing import Literal

import numpy as np

from .transformation import Transformation


@dataclass(frozen=True)
class NormalDistribution3d:
    mean: np.ndarray[tuple[Literal[3]], np.dtype[np.float32]]
    covariance: np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.float32]]

    def transform(self, transformation: Transformation) -> 'NormalDistribution3d':
        rotation = transformation.rotation
        translation = transformation.translation

        mean = translation + rotation @ self.mean
        covariance = rotation @ self.covariance @ rotation.T

        return NormalDistribution3d(mean, covariance)

    def sample(self, samples: int): ...
