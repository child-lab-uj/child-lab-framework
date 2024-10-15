import typing
from dataclasses import dataclass

import cv2
import numpy as np

from ....typing.array import FloatArray1, FloatArray2


@dataclass(repr=False, frozen=True)
class Result:
    rotation: FloatArray1
    translation: FloatArray1
    intrinsics: FloatArray2
    distortion: FloatArray1

    def __repr__(self) -> str:
        rotation = self.rotation
        translation = self.translation
        return f'Result:\n{translation = }\n{rotation = }'

    # Use np.einsum('ij,kmj->kmi') for more sophisticated transformations
    # (i.e. on n_people x n_features x n_coordinates arrays)
    def project(self, points: FloatArray2) -> FloatArray2:
        result = np.squeeze(
            cv2.projectPoints(
                points,
                self.rotation,
                self.translation,
                self.intrinsics,
                self.distortion,
            )[0]
        )

        return typing.cast(FloatArray2, result)
