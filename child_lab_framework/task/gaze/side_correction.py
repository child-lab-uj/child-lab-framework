from typing import Never
import numpy as np

from .. import pose, face
from ..pose.keypoint import YoloKeypoint
from ..face import Eye
from ...core.algebra import rotation_matrix, Axis
from ...core.video import Perspective, Properties
from ...core.sequence import (imputed_with_reference_inplace, imputed_with_zeros_reference_inplace)
from ...typing.array import FloatArray1, FloatArray2, FloatArray3


def anchor_person_index(
    n_people: int,
    perspective: Perspective
) -> int:
    match perspective:
        case Perspective.WINDOW_LEFT:
            return n_people - 1

        case Perspective.WINDOW_RIGHT:
            return 0

    return Never


# NOTE: assumes matched person indices in both views
def transformation(
    ceiling_shoulders: FloatArray3,
    side_shoulders: FloatArray3,
    perspective: Perspective
) -> tuple[FloatArray2, FloatArray1]:
    anchor = anchor_person_index(
        min(len(ceiling_shoulders), len(side_shoulders)),
        perspective
    )

    anchor_ceiling_shoulders: FloatArray2 = ceiling_shoulders[anchor, ...]
    anchor_side_shoulders: FloatArray2 = side_shoulders[anchor, ...]

    anchor_ceiling_shoulder_components: FloatArray1 = anchor_ceiling_shoulders[1, :-1] - anchor_ceiling_shoulders[0, :-1]
    anchor_side_shoulder_components: FloatArray1 = anchor_side_shoulders[1, :-1] - anchor_side_shoulders[0, :-1]

    print(f'\n{anchor_ceiling_shoulder_components = }')
    print(f'\n{anchor_side_shoulder_components = }')

    alpha, gamma = np.arccos(anchor_side_shoulder_components / anchor_ceiling_shoulder_components)
    print(f'\n{alpha = }, {gamma = }\n')

    rotation: FloatArray2 = rotation_matrix(alpha, Axis.X) @ rotation_matrix(gamma, Axis.Z)

    anchor_side_shoulders[:, 2] = 1.0

    anchor_ceiling_shoulder_components_reconstructed: FloatArray2 = anchor_side_shoulders @ rotation
    translation_components: FloatArray2 = anchor_ceiling_shoulder_components_reconstructed - anchor_ceiling_shoulder_components.reshape(-1, 1)

    translation: FloatArray1 = (translation_components[0, :] + translation_components[1, :]) / 2.0
    translation = translation.reshape(-1, 1)

    return rotation, translation


def eye_normals(eyes: FloatArray3) -> FloatArray2:
    vecs1: FloatArray2 = eyes[:, 0, :] - eyes[:, 1, :]  # top -> right
    vecs2: FloatArray2 = eyes[:, 2, :] - eyes[:, 1, :]  # top -> left

    return np.cross(vecs1, vecs2)


# TODO: Add Iris estimation and more sophisticated algebra (cv2.solvePnP...)
# NOTE: assumes person-matched poses and faces
def estimate(
    ceiling_poses: list[pose.Result | None],
    side_poses: list[pose.Result | None],
    side_faces: list[face.Result | None],
    perspective: Perspective,
    ceiling: Properties,
    side: Properties
) -> FloatArray3 | None:
    results: list[FloatArray2 | None] = []

    for ceiling_pose, side_pose, faces in zip(ceiling_poses, side_poses, side_faces):
        if ceiling_pose is None or side_pose is None or faces is None:
            results.append(None)
            continue

        rotation, translation = transformation(
            ceiling_pose.keypoints.view()[:, [YoloKeypoint.LEFT_SHOULDER, YoloKeypoint.RIGHT_SHOULDER], :],
            side_pose.keypoints.view()[:, [YoloKeypoint.LEFT_SHOULDER, YoloKeypoint.RIGHT_SHOULDER], :],
            perspective
        )

        right_eyes_normals = eye_normals(faces.eyes(Eye.Right)[:, :, :3])
        ceiling_gazes = (right_eyes_normals @ rotation + translation.T).view()[:, [0, 1]]

        results.append(ceiling_gazes)

    results_imputed = imputed_with_zeros_reference_inplace(results)

    if results_imputed is None:
        return None

    return np.stack(results_imputed)
