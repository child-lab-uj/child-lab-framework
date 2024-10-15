import numpy as np

from ...core.sequence import imputed_with_zeros_reference_inplace
from ...typing.array import FloatArray2, FloatArray3
from .. import face, pose
from ..camera.transformation import Result as Transformation
from ..face import Eye


def eye_normals(eyes: FloatArray3) -> FloatArray2:
    vecs1: FloatArray2 = eyes[:, 0, :] - eyes[:, 1, :]  # top -> right
    vecs2: FloatArray2 = eyes[:, 2, :] - eyes[:, 1, :]  # top -> left

    return np.cross(vecs1, vecs2)


# TODO: Add Iris estimation and more sophisticated algebra (cv2.solvePnP, OpenFace-inspired calculations etc.)
# NOTE: assumes person-matched poses and faces
def estimate(
    ceiling_poses: list[pose.Result | None],
    side_poses: list[pose.Result | None],
    side_faces: list[face.Result | None],
    side_to_ceiling_transformations: list[Transformation | None],
) -> FloatArray3 | None:
    results: list[FloatArray2 | None] = []

    for ceiling_pose, side_pose, faces, transformation in zip(
        ceiling_poses, side_poses, side_faces, side_to_ceiling_transformations
    ):
        if (
            ceiling_pose is None
            or side_pose is None
            or faces is None
            or transformation is None
        ):
            results.append(None)
            continue

        right_eyes_normals = eye_normals(faces.eyes(Eye.Right)[:, :, :3])
        ceiling_gazes = transformation.project(right_eyes_normals).view()[:, [0, 1]]

        results.append(ceiling_gazes)

    results_imputed = imputed_with_zeros_reference_inplace(results)

    if results_imputed is None:
        return None

    return np.stack(results_imputed)
