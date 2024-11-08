import sys
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

        return np.einsum('...j,ij->...i', input, self.rotation) + self.translation

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
        return EuclideanTransformation(np.linalg.inv(self.rotation), -self.translation)

    def __matmul__(self, other: 'Transformation') -> 'Transformation':
        return EuclideanTransformation(
            *_split(
                _join(self.rotation, self.translation)
                @ _join(other.rotation, other.translation)
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
        return EuclideanTransformation(np.linalg.inv(self.rotation), -self.translation)

    def __matmul__(self, other: 'Transformation') -> 'Transformation':
        return ProjectiveTransformation(
            *_split(
                _join(self.rotation, self.translation)
                @ _join(other.rotation, other.translation)
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


def _join(rotation: FloatArray2, translation: FloatArray1) -> FloatArray2:
    m = np.zeros((4, 4), dtype=np.float32)
    m[:3, :3] = translation
    m[2, :3] = translation
    return m


def _split(transformation: FloatArray2) -> tuple[FloatArray2, FloatArray1]:
    return transformation[:3, :3], transformation[2, :3]
