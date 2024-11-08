from dataclasses import dataclass
from typing import Self

import numpy as np

from ..typing.array import FloatArray1, FloatArray2, FloatArray3
from . import serialization


@dataclass(unsafe_hash=True, frozen=True, repr=False)
class Calibration:
    optical_center: tuple[float, float]
    focal_length: tuple[float, float]
    distortion: FloatArray1

    @classmethod
    def heuristic(cls, height: int, width: int) -> Self:
        cx = width / 2.0
        cy = height / 2.0

        fx = 500.0 * width / 640.0
        fy = 500.0 * height / 480.0
        fx = (fx + fy) / 2.0
        fy = fx

        distortion = np.zeros(5, dtype=np.float32)

        return cls((cx, cy), (fx, fy), distortion)

    # @lru_cache(1)
    def resized(self, width_scale: float, height_scale: float) -> 'Calibration':
        cx, cy = self.optical_center
        fx, fy = self.focal_length

        return Calibration(
            (cx * width_scale, cy * height_scale),
            (fx * width_scale, fy * height_scale),
            self.distortion,
        )

    # @lru_cache(1)
    def flat(self) -> tuple[float, float, float, float]:
        return (*self.focal_length, *self.optical_center)

    @property
    def intrinsics(self) -> FloatArray2:
        m = np.zeros((3, 3), dtype=np.float32)
        m[0, 0], m[1, 1] = self.focal_length
        m[0:2, 2] = self.optical_center
        m[2, 2] = 1.0

        return m

    def depth_to_3d(self, depth: FloatArray2) -> FloatArray3:
        u = np.arange(depth.shape[1])
        v = np.arange(depth.shape[0])
        u, v = np.meshgrid(u, v)

        x = (u - self.optical_center[0]) * depth / self.focal_length[0]
        y = (v - self.optical_center[1]) * depth / self.focal_length[1]
        z = depth

        # Expand dimensions to make it 3D
        x = np.expand_dims(x, axis=-1)
        y = np.expand_dims(y, axis=-1)
        z = np.expand_dims(z, axis=-1)

        return np.concatenate((x, y, z), axis=-1)

    def serialize(self) -> dict[str, serialization.Value]:
        return {
            'optical_center': list(self.optical_center),
            'focal_length': list(self.focal_length),
            'distortion': self.distortion.tolist(),
        }

    @classmethod
    def deserialize(cls, data: dict[str, serialization.Value]) -> Self:
        match data:
            case {
                'optical_center': [float(cx), float(cy)],
                'focal_length': [float(fx), float(fy)],
                'distortion': [float(d1), float(d2), float(d3), float(d4), float(d5)],
                **_other,
            }:
                optical_center = cx, cy
                focal_length = fx, fy
                distortion = np.array([d1, d2, d3, d4, d5], dtype=np.float32)
                return cls(optical_center, focal_length, distortion)

            case other:
                raise serialization.DeserializeError(
                    'Expected dictionary with: '
                    'optical_center: tuple[float, float], '
                    'focal_length: tuple[float, float], '
                    'and distortion: list[float] (shape: 5), '
                    f'got {other}',
                )

    def __repr__(self) -> str:
        cx, cy = self.optical_center
        fx, fy = self.focal_length

        return (
            'Optical center:\n'
            f'  x: {cx:.2f},\n'
            f'  y: {cy:.2f},\n'
            'Focal length:\n'
            f'  x: {fx:.2f},\n'
            f'  y: {fy:.2f}'
        )
