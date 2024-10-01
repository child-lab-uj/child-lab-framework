import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat, starmap

import numpy as np

from ...core.video import Properties
from ...typing.array import FloatArray2
from ...typing.stream import Fiber
from ...typing.transformation import Transformation
from .. import face, pose
from ..camera.transformation import heuristic
from . import ceiling_baseline
from . import mini_face as mf

# Multi-camera gaze direction estimation without strict algebraic camera models:
# 1. estimate actor's skeleton on each frame in both cameras
#    (heuristic: adults' keypoints have higher variance, children have smaller bounding boxes)
# 2. compute a ceiling baseline vector (perpendicular to shoulder line in a ceiling camera)
# 3. detect actor's face on the other camera
# 4. compute the offset-baseline vector (normal to face, Wim's MediaPipe solution')
# 5. Rotate it to the celing camera's space and combine with the ceiling baseline


type Input = tuple[
	list[pose.Result | None] | None,
	list[pose.Result | None] | None,
	list[pose.Result | None] | None,
	list[face.Result | None] | None,
	list[face.Result | None] | None,
	list[heuristic.Result | None] | None,
	list[heuristic.Result | None] | None,
]


@dataclass
class Result:
	centres: FloatArray2
	directions: FloatArray2


class Estimator:
	BASELINE_WEIGHT: float = 0.0
	COLLECTIVE_CORRECTION_WEIGHT: float = 1.0 - BASELINE_WEIGHT

	ceiling_properties: Properties
	window_left_properties: Properties
	window_right_properties: Properties

	executor: ThreadPoolExecutor

	def __init__(
		self,
		executor: ThreadPoolExecutor,
		ceiling_properties: Properties,
		window_left_properties: Properties,
		window_right_properties: Properties,
	) -> None:
		self.executor = executor
		self.ceiling_properties = ceiling_properties
		self.window_left_properties = window_left_properties
		self.window_right_properties = window_right_properties

	def predict(
		self,
		ceiling_pose: pose.Result,
		window_left_gaze: mf.Result | None,
		window_right_gaze: mf.Result | None,
		window_left_to_ceiling: Transformation | None,
		window_right_to_ceiling: Transformation | None,
	) -> Result:
		centres, directions = ceiling_baseline.estimate(ceiling_pose)

		correct_from_left = (
			window_left_gaze is not None and window_left_to_ceiling is not None
		)

		correct_from_right = (
			window_right_gaze is not None and window_right_to_ceiling is not None
		)

		correction_count = int(correct_from_left) + int(correct_from_right)

		if correction_count == 0:
			return Result(centres, directions)

		correction_weight = self.COLLECTIVE_CORRECTION_WEIGHT / float(correction_count)
		baseline_weight = self.BASELINE_WEIGHT

		if correction_count > 0.0:
			centres *= baseline_weight
			directions *= baseline_weight

		# Use np.einsum('ij,kmj->kmi') for transformation without simplification (i.e. on n_people x 2 x 3 arrays)
		# cannot reuse `correct_from_left` because the type checker gets confused
		if window_left_gaze is not None and window_left_to_ceiling is not None:
			rotation_transposed = window_left_to_ceiling.rotation.T
			translation = window_left_to_ceiling.translation.T

			eyes_simplified = np.squeeze(np.mean(window_left_gaze.eyes, axis=1))

			left_centre_correction: FloatArray2 = (
				eyes_simplified @ rotation_transposed + translation
			).view()[:, :2]

			directions_simplified = np.squeeze(
				np.mean(window_left_gaze.directions, axis=1)
			)

			left_direction_correction: FloatArray2 = (
				directions_simplified @ rotation_transposed + translation
			).view()[:, :2]

			centres += correction_weight * left_centre_correction
			directions += correction_weight * left_direction_correction

		if window_right_gaze is not None and window_right_to_ceiling is not None:
			rotation_transposed = window_right_to_ceiling.rotation.T
			translation = window_right_to_ceiling.translation.T

			eyes_simplified = np.squeeze(np.mean(window_right_gaze.eyes, axis=1))

			right_centre_correction: FloatArray2 = (
				eyes_simplified @ rotation_transposed + translation
			).view()[:, :2]

			directions_simplified = np.squeeze(
				np.mean(window_right_gaze.directions, axis=1)
			)

			right_direction_correction: FloatArray2 = (
				directions_simplified @ rotation_transposed + translation
			).view()[:, :2]

			centres += correction_weight * right_centre_correction
			directions += correction_weight * right_direction_correction

		return Result(centres, directions)

	def __predict_safe(
		self,
		ceiling_pose: pose.Result | None,
		window_left_gaze: mf.Result | None,
		window_right_gaze: mf.Result | None,
		window_left_to_ceiling: Transformation | None,
		window_right_to_ceiling: Transformation | None,
	) -> Result | None:
		if ceiling_pose is None:
			return None

		return self.predict(
			ceiling_pose,
			window_left_gaze,
			window_right_gaze,
			window_left_to_ceiling,
			window_right_to_ceiling,
		)

	# NOTE: heuristic idea: actors seen from right and left are in reversed lexicographic order
	async def stream(self) -> Fiber[Input | None, list[Result | None] | None]:
		executor = self.executor
		loop = asyncio.get_running_loop()

		results: list[Result | None] | None = None

		while True:
			match (yield results):
				case (
					list(ceiling_pose),
					window_left_gaze,
					window_right_gaze,
					window_left_to_ceiling,
					window_right_to_ceiling,
				):
					results = await loop.run_in_executor(
						executor,
						lambda: list(
							starmap(
								self.__predict_safe,
								zip(
									ceiling_pose,
									window_left_gaze or repeat(None),
									window_right_gaze or repeat(None),
									window_left_to_ceiling or repeat(None),
									window_right_to_ceiling or repeat(None),
								),
							)
						),
					)

				case _:
					results = None
