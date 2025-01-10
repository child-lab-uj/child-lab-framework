import sys
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Self, TypeVar

import cv2
import numpy as np

from ...typing.array import FloatArray1, FloatArray2
from .. import serialization
from ..calibration import Calibration

Shape = TypeVar('Shape', bound=tuple)


class Transformation(ABC):
    rotation: FloatArray2
    translation: FloatArray1

    def transform[Shape, DataType: Any](
        self,
        input: np.ndarray[Shape, DataType],
    ) -> np.ndarray[Shape, DataType]:
        if len(input.shape) == 1:
            return self.rotation @ input + self.translation  # type: ignore

        translation = np.squeeze(self.translation)  # TODO: get rid of this operation
        return np.einsum('...j,ij->...i', input, self.rotation) + translation

    @property
    @abstractmethod
    def inverse(self) -> 'Transformation': ...

    @abstractmethod
    def __matmul__(self, other: 'Transformation') -> 'Transformation': ...

    @abstractmethod
    def serialize(self) -> dict[str, serialization.Value]: ...

    @classmethod
    def deserialize(cls, data: dict[str, serialization.Value]) -> 'Transformation | None':
        subclass_name = data.get('type')

        if not isinstance(subclass_name, str):
            raise serialization.DeserializeError(
                f'Expected "type" field to contain string, got {type(subclass_name)}'
            )

        module = sys.modules[__name__]

        if not hasattr(module, subclass_name):
            raise serialization.DeserializeError(
                f'Failed to retrieve class "{subclass_name}" in module {module}'
            )

        subclass = getattr(module, subclass_name)

        if not issubclass(subclass, Transformation):
            raise serialization.DeserializeError(
                f'Expected {subclass_name} to subtype Transformation, got {type(subclass)}'
            )

        return subclass.deserialize(data)


@dataclass(frozen=True)
class EuclideanTransformation(Transformation):
    rotation: FloatArray2
    translation: FloatArray1

    @property
    def inverse(self) -> Transformation:
        inverse_rotation = self.rotation.transpose()
        inverse_translation = -inverse_rotation @ self.translation
        return EuclideanTransformation(inverse_rotation, inverse_translation)

    def __matmul__(self, other: 'Transformation') -> 'Transformation':
        return EuclideanTransformation(
            *_combine(
                other.rotation,
                other.translation,
                self.rotation,
                self.translation,
            )
        )

    def serialize(self) -> dict[str, serialization.Value]:
        return {
            'type': 'EuclideanTransformation',
            'rotation': self.rotation.tolist(),
            'translation': self.translation.tolist(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, serialization.Value]) -> Self | None:
        match data:
            case {
                'type': 'EuclideanTransformation',
                'rotation': [
                    [float(r00), float(r01), float(r02)],
                    [float(r10), float(r11), float(r12)],
                    [float(r20), float(r21), float(r22)],
                ],
                'translation': [float(t0), float(t1), float(t2)],
                **_rest,
            }:
                return cls(
                    np.array(
                        [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]],
                        dtype=np.float32,
                    ),
                    np.array([t0, t1, t2], dtype=np.float32),
                )

            case other:
                raise serialization.DeserializeError(
                    'Expected dictionary with: '
                    'type: str, '
                    'rotation: list[list[float]] (shape: 3 x 3), '
                    'and translation: list[float] (shape: 3) '
                    f'keys, got {other}'
                )


@dataclass(frozen=True)
class ProjectiveTransformation(Transformation):
    rotation: FloatArray2
    translation: FloatArray1
    calibration: Calibration

    def project[Shape, DataType: Any](
        self,
        input: np.ndarray[Shape, DataType],
    ) -> np.ndarray[Shape, DataType]:
        calibration = self.calibration

        if input.ndim <= 2:
            return np.squeeze(
                cv2.projectPoints(
                    input,
                    self.rotation,
                    self.translation,
                    calibration.intrinsics,
                    calibration.distortion,
                )[0]
            )  # type: ignore

        # TODO: implement a general projection and undistortion formula using Einstein notation
        raise NotImplementedError()

    @property
    def inverse(self) -> Transformation:
        return EuclideanTransformation(self.rotation, self.translation).inverse

    def __matmul__(self, other: 'Transformation') -> 'Transformation':
        return ProjectiveTransformation(
            *_combine(
                other.rotation,
                other.translation,
                self.rotation,
                self.translation,
            ),
            self.calibration,
        )

    def serialize(self) -> dict[str, serialization.Value]:
        return {
            'type': 'ProjectiveTransformation',
            'rotation': self.rotation.tolist(),
            'translation': self.translation.tolist(),
            'calibration': self.calibration.serialize(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, serialization.Value]) -> Self | None:
        match data:
            case {
                'type': 'ProjectiveTransformation',
                'rotation': [
                    [float(r00), float(r01), float(r02)],
                    [float(r10), float(r11), float(r12)],
                    [float(r20), float(r21), float(r22)],
                ],
                'translation': [float(t0), float(t1), float(t2)],
                'calibration': dict(maybe_calibration),
                **_rest,
            }:
                calibration = Calibration.deserialize(maybe_calibration)

                if calibration is None:
                    return None

                return cls(
                    np.array(
                        [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]],
                        dtype=np.float32,
                    ),
                    np.array([t0, t1, t2], dtype=np.float32),
                    calibration,
                )

            case other:
                raise serialization.DeserializeError(
                    'Expected dictionary with: '
                    'type: str, '
                    'rotation: list[list[float]] (shape: 3 x 3), '
                    'translation: list[float] (shape: 3) '
                    'and calibration: Calibration '
                    f'keys, got {other}',
                )


def _combine(
    inner_rotation: FloatArray2,
    inner_translation: FloatArray1,
    outer_rotation: FloatArray2,
    outer_translation: FloatArray1,
) -> tuple[FloatArray2, FloatArray1]:
    """
    Chain two transformations (given as rotations and translations) to produce a new transformation: `x -> outer_transformation(inner_transformation(x))`.

    Returns
    ---
    result: tuple[FloatArray2, FloatArray1]
    """

    inner_rotation_vector, _ = cv2.Rodrigues(inner_rotation)
    outer_rotation_vector, _ = cv2.Rodrigues(inner_rotation)

    combined_rotation_vector, combined_translation, *_ = cv2.composeRT(
        inner_rotation_vector,
        inner_translation,
        outer_rotation_vector,
        outer_translation,
    )

    combined_rotation, _ = cv2.Rodrigues(combined_rotation_vector)
    combined_translation = np.squeeze(combined_translation)

    return (
        typing.cast(FloatArray2, combined_rotation),
        typing.cast(FloatArray1, combined_translation),
    )
