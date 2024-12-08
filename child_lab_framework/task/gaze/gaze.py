import asyncio
import typing
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from itertools import repeat, starmap

import cv2
import mini_face as mf
import numpy as np
import scipy

from ...core.algebra import normalized_3d
from ...core.calibration import Calibration
from ...core.stream import InvalidArgumentException
from ...core.transformation import Transformation
from ...core.video import Frame, Properties
from ...postprocessing.imputation import (
    imputed_with_zeros_reference,
)
from ...typing.array import (
    BoolArray1,
    BoolArray2,
    FloatArray1,
    FloatArray2,
    FloatArray3,
    IntArray1,
)
from ...typing.stream import Fiber
from ...util import MODELS_DIR as MODELS_ROOT
from .. import face, visualization


@dataclass(frozen=True)
class Result:
    eyes: FloatArray3
    directions: FloatArray3


@dataclass(frozen=True)
class Result3d:
    eyes: FloatArray3
    directions: FloatArray3

    # For numerical stability during transformation of a normalized `directions`
    __STABILIZING_MULTIPLIER: typing.ClassVar[float] = 100.0

    @classmethod
    def approximated(
        cls,
        results: Sequence['Result3d | None'],
    ) -> list['Result3d'] | None:
        if all(item is not None for item in results):
            return typing.cast(list[Result3d], list(results))

        if all(item is None for item in results):
            return None

        n_time_points = len(results)
        time: FloatArray1 = np.zeros(n_time_points, dtype=np.float32)

        max_actors = max(len(result.eyes) for result in results if result is not None)

        actor_occurrences_in_time: BoolArray2 = np.zeros(
            (max_actors, n_time_points),
            dtype=bool,
        )
        actor_results_in_time: FloatArray3 = np.zeros(
            (max_actors, n_time_points, 12),
            dtype=np.float32,
        )

        for i, result in enumerate(results):
            if result is None:
                continue

            eyes = result.__unbound_actor_eyes
            directions = result.__unbound_actor_directions

            n_actors = len(eyes)

            actor_occurrences_in_time[:n_actors, i] = True
            np.copyto(actor_results_in_time[:n_actors, i, :6], eyes)
            np.copyto(actor_results_in_time[:n_actors, i, 6:], directions)

        actor_time_mask: BoolArray1
        for actor, actor_time_mask in enumerate(actor_occurrences_in_time):
            weights = np.ones(n_time_points, dtype=np.float32)
            weights[~actor_time_mask] = 0.0

            n_known_points = int(actor_time_mask.sum())
            degree = n_known_points // 2

            coefficients = np.polyfit(
                time,
                actor_results_in_time[actor, ...],
                deg=degree,
                w=weights,
            )

            for feature in range(12):
                polynomial = np.polynomial.Polynomial(coefficients[:, feature])
                actor_results_in_time[actor, :, feature] = polynomial(time)

        return list(
            map(
                Result3d.__from_flat,
                np.transpose(actor_results_in_time, (1, 0, 2)),
            )
        )

    # TODO: Deduplicate the boilerplate
    @classmethod
    def interpolated(
        cls,
        results: Sequence['Result3d | None'],
    ) -> list['Result3d'] | None:
        if all(item is not None for item in results):
            return typing.cast(list[Result3d], list(results))

        if all(item is None for item in results):
            return None

        n_time_points = len(results)
        time: FloatArray1 = np.zeros(n_time_points, dtype=np.float32)

        max_actors = max(len(result.eyes) for result in results if result is not None)

        actor_occurrences_in_time: BoolArray2 = np.zeros(
            (max_actors, n_time_points),
            dtype=bool,
        )
        actor_results_in_time: FloatArray3 = np.zeros(
            (max_actors, n_time_points, 12),
            dtype=np.float32,
        )

        for i, result in enumerate(results):
            if result is None:
                continue

            eyes = result.__unbound_actor_eyes
            directions = result.__unbound_actor_directions

            n_actors = len(eyes)

            actor_occurrences_in_time[:n_actors, i] = True
            np.copyto(actor_results_in_time[:n_actors, i, :6], eyes)
            np.copyto(actor_results_in_time[:n_actors, i, 6:], directions)

        for actor, actor_time_mask in enumerate(actor_occurrences_in_time):
            actor_results: FloatArray2 = actor_results_in_time[actor, actor_time_mask, :]

            actor_known_time = time[actor_time_mask]
            negative_time_mask = ~actor_time_mask
            actor_unknown_time = time[negative_time_mask]

            interpolated_values = np.squeeze(
                scipy.interpolate.krogh_interpolate(
                    actor_known_time, actor_results, actor_unknown_time
                )
            )

            actor_results_in_time[actor, negative_time_mask, ...] = interpolated_values

        return list(
            map(
                Result3d.__from_flat,
                np.transpose(actor_results_in_time, (1, 0, 2)),
            )
        )

    @cached_property
    def __unbound_actor_eyes(self) -> list[FloatArray1]:
        return list(self.eyes.reshape(-1, 6))

    @cached_property
    def __unbound_actor_directions(self) -> list[FloatArray1]:
        return list(self.directions.reshape(-1, 6))

    @cached_property
    def flat(self) -> FloatArray1:
        return np.concatenate((self.eyes.flatten(), self.directions.flatten()))

    @classmethod
    def __from_flat(cls, data: FloatArray2) -> typing.Self:
        assert data.ndim == 2

        n_rows, n_cols = data.shape
        assert n_rows > 0
        assert n_cols == 12

        eyes = data[:, :6]
        directions = data[:, 6:]

        return cls(
            eyes.reshape(-1, 2, 3),
            directions.reshape(-1, 2, 3),
        )

    @property
    def flat_points(self) -> FloatArray2:
        return self.__flat_points

    @cached_property
    def __flat_points(self) -> FloatArray2:
        return np.concatenate(self.eyes)

    def project(self, calibration: Calibration) -> Result:
        cx, cy = calibration.optical_center
        fx, fy = calibration.focal_length

        # TODO: remove the rescale (Issue #60)
        eyes = self.eyes.copy() * 8.0 / 28.0
        z = eyes[..., -1]
        eyes[..., 0] *= fx
        eyes[..., 1] *= fy
        eyes[..., 0] /= z
        eyes[..., 1] /= z
        eyes[..., 0] += cx
        eyes[..., 1] += cy

        ends = self.eyes + self.__STABILIZING_MULTIPLIER * self.directions
        z = ends[..., -1]
        ends[..., 0] *= fx
        ends[..., 1] *= fy
        ends[..., 0] /= z
        ends[..., 1] /= z
        ends[..., 0] += cx
        ends[..., 1] += cy

        directions = ends - eyes
        directions_2d = directions[..., :-1]
        directions_2d = normalized_3d(directions_2d)

        return Result(eyes[..., :-1], directions_2d)

    def transform(self, transformation: Transformation) -> 'Result3d':
        multiplier = self.__STABILIZING_MULTIPLIER

        return Result3d(
            transformation.transform(self.eyes),
            transformation.transform(multiplier * self.directions) / multiplier,
        )

    # TODO: Move to `Result` for consistency with other models
    def visualize(
        self,
        frame: Frame,
        frame_properties: Properties,
        configuration: visualization.Configuration,
    ) -> Frame:
        if not configuration.gaze_draw_lines:
            return frame

        color = configuration.gaze_line_color
        thickness = configuration.gaze_line_thickness
        line_length = configuration.gaze_line_length

        projected = self.project(frame_properties.calibration)
        starts = projected.eyes
        ends = starts + float(line_length) * projected.directions

        actor_starts: FloatArray2
        actor_ends: FloatArray2

        for actor_starts, actor_ends in zip(starts, ends):
            cv2.line(
                frame,
                typing.cast(cv2.typing.Point, actor_starts[0, :2].astype(np.int32)),
                typing.cast(cv2.typing.Point, actor_ends[0, :2].astype(np.int32)),
                color,
                thickness,
            )

            cv2.line(
                frame,
                typing.cast(cv2.typing.Point, actor_starts[1, :2].astype(np.int32)),
                typing.cast(cv2.typing.Point, actor_ends[1, :2].astype(np.int32)),
                color,
                thickness,
            )

        return frame


type Input = tuple[list[Frame], list[face.Result | None] | None]


class Estimator:
    MODELS_DIR = MODELS_ROOT / 'model'

    input: Properties

    optical_center_x: float
    optical_center_y: float

    extractor: mf.gaze.Extractor

    executor: ThreadPoolExecutor | None

    def __init__(
        self,
        *,
        input: Properties,
        wild: bool = False,
        multiple_views: bool = False,
        limit_angles: bool = False,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.executor = executor

        self.input = input

        calibration = input.calibration
        self.optical_center_x, self.optical_center_y = calibration.optical_center

        self.extractor = mf.gaze.Extractor(
            mode=mf.PredictionMode.VIDEO,
            focal_length=calibration.focal_length,
            optical_center=calibration.optical_center,
            fps=input.fps,
            models_directory=self.MODELS_DIR,
        )

    def predict(self, frame: Frame, faces: face.Result) -> Result3d | None:
        extractor = self.extractor

        eyes: list[FloatArray3 | None] = []
        directions: list[FloatArray3 | None] = []  # in fact arrays are 1 x 2 x 3

        box: IntArray1

        for box in faces.boxes:
            region = box.astype(np.uint32)
            region[2] -= region[0]
            region[3] -= region[1]

            detection = extractor.predict(frame, region)

            if detection is None:
                eyes.append(None)
                directions.append(None)
                continue

            eyes.append(detection.eyes)  # type: ignore
            directions.append(detection.directions)  # type: ignore

        match (
            imputed_with_zeros_reference(eyes),
            imputed_with_zeros_reference(directions),
        ):
            case list(eyes_imputed), list(directions_imputed):
                return Result3d(
                    np.concatenate(eyes_imputed, axis=0),
                    np.concatenate(directions_imputed, axis=0),
                )

            case _:
                return None

    def predict_batch(
        self,
        frames: list[Frame],
        faces: list[face.Result],
    ) -> list[Result3d | None]:
        return list(starmap(self.predict, zip(frames, faces)))

    def __predict_safe(self, frame: Frame, faces: face.Result | None) -> Result3d | None:
        if faces is None:
            return None

        return self.predict(frame, faces)

    async def stream(self) -> Fiber[Input | None, list[Result3d | None] | None]:
        executor = self.executor
        if executor is None:
            raise RuntimeError(
                'Processing in the stream mode requires the Estimator to have an executor. Please pass an "executor" argument to the estimator constructor'
            )

        loop = asyncio.get_running_loop()

        results: list[Result3d | None] | None = None

        while True:
            match (yield results):
                case list(frames), faces:
                    results = await loop.run_in_executor(
                        executor,
                        lambda: list(
                            starmap(
                                self.__predict_safe, zip(frames, faces or repeat(None))
                            )
                        ),
                    )

                case None:
                    results = None

                case _:
                    raise InvalidArgumentException()
