from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from ...core import video
from ...core.algebra import rotation_matrix_between_vectors
from ...core.probability import NormalDistribution3d
from ...core.transformation import EuclideanTransformation, Transformation
from ...typing.array import FloatArray1, FloatArray2
from .. import gaze, visualization

BASELINE_GAZE_DIRECTION = np.array([1.0, 0.0, 0.0], dtype=np.float32)
BASELINE_CENTRE = np.array([0.0, 0.0, 0.0], dtype=np.float32)
BASELINE_COVARIANCE = (
    np.array(
        [
            [5.0, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.2],
        ],
        dtype=np.float32,
    )
    * 100.0
)


@dataclass(frozen=True)
class RigidAttentionModel:
    gaze_direction: np.ndarray[tuple[Literal[3]], np.dtype[np.float32]] = field(
        default_factory=lambda: BASELINE_GAZE_DIRECTION
    )

    distribution: NormalDistribution3d = field(
        default_factory=lambda: NormalDistribution3d(
            BASELINE_CENTRE,
            BASELINE_COVARIANCE,
        )
    )


@dataclass(frozen=True)
class Result3d:
    distributions: list[NormalDistribution3d | None]

    def transform(self, transformation: Transformation) -> 'Result3d':
        return Result3d(
            [
                distribution.transform(transformation)
                if distribution is not None
                else None
                for distribution in self.distributions
            ]
        )

    def visualize(
        self,
        frame: video.Frame,
        frame_properties: video.Properties,
        configuration: visualization.Configuration,
    ) -> video.Frame:
        _threshold = configuration.joint_attention_confidence_threshold
        n_samples = configuration.joint_attention_distribution_samples

        calibration = frame_properties.calibration
        fx, fy = calibration.focal_length
        cx, cy = calibration.optical_center

        rng = np.random.default_rng()

        x: FloatArray1
        y: FloatArray1
        z: FloatArray1

        # point: IntArray1

        for distribution in self.distributions:
            if distribution is None:
                continue

            samples = rng.multivariate_normal(
                distribution.mean,
                distribution.covariance,
                n_samples,
            ).astype(np.float32)

            x = samples[..., 0]
            y = samples[..., 1]
            z = samples[..., 2]

            x[...] = x * fx / z + cx
            y[...] = y * fy / z + cy

            if np.any(y <= 0):
                y *= -1.0

            for point in zip(x.astype(int), y.astype(int)):
                cv2.circle(
                    frame,
                    point,  # type: ignore
                    3,
                    (255.0, 0.0, 255.0, 1.0),
                    1,
                )

        return frame


@dataclass(frozen=True)
class Result: ...  # TODO: Come out with a method to project the 3D normal distribution to a 2D plane


class Estimator:
    gaze_model: RigidAttentionModel

    def __init__(
        self,
        gaze_model: RigidAttentionModel | None = None,
    ) -> None:
        self.gaze_model = gaze_model if gaze_model is not None else RigidAttentionModel()

    def predict(self, gazes: gaze.Result3d) -> Result3d:
        gaze_model = self.gaze_model
        rigid_direction = gaze_model.gaze_direction
        rigid_distribution = gaze_model.distribution

        simplified_directions = np.mean(gazes.directions, axis=1)

        distribution_rotations = [
            rotation_matrix_between_vectors(rigid_direction, actor_direction)
            for actor_direction in simplified_directions
        ]

        distribution_means: FloatArray2 = np.mean(gazes.eyes, axis=1)

        return Result3d(
            [
                rigid_distribution.transform(
                    EuclideanTransformation(rotation, translation)  # type: ignore
                )
                if rotation is not None
                else None
                for rotation, translation in zip(
                    distribution_rotations, distribution_means
                )
            ]
        )

    def predict_batch(
        self,
        gazes: list[gaze.Result3d],
    ) -> list[Result3d]:
        return [self.predict(result) for result in gazes]
