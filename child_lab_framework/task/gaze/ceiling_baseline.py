from ...core.algebra import normalized, orthogonal
from ...task import pose
from ...typing.array import FloatArray2
from ..pose.keypoint import YoloKeypoint


def estimate(poses: pose.Result) -> tuple[FloatArray2, FloatArray2]:
	left_shoulder: FloatArray2 = poses.keypoints[:, YoloKeypoint.LEFT_SHOULDER.value, :2]
	right_shoulder: FloatArray2 = poses.keypoints[
		:, YoloKeypoint.RIGHT_SHOULDER.value, :2
	]

	centres: FloatArray2 = (left_shoulder + right_shoulder) / 2.0

	# convention: shoulder vector goes from left to right -> versor (calculated as [y, -x]) points to the actor's front
	directions = normalized(orthogonal(right_shoulder - left_shoulder))

	return centres, directions
