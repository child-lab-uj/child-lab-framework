import numpy as np

from .....core.algebra import Axis, rotation_matrix
from .....core.video import Perspective
from .....typing.array import FloatArray1, FloatArray2, FloatArray3


# Trivial at the moment
def anchor_person_index(n_people: int, perspective: Perspective) -> int:
    match perspective:
        case Perspective.WINDOW_LEFT:
            return n_people - 1

        case Perspective.WINDOW_RIGHT:
            return 0

        case _:
            assert False, 'unreachable'


# TODO: Add depth
# NOTE: assumes matched person indices in both views
def estimate(
    ceiling_shoulders: FloatArray3, side_shoulders: FloatArray3, perspective: Perspective
) -> tuple[FloatArray2, FloatArray1]:
    anchor = anchor_person_index(
        min(len(ceiling_shoulders), len(side_shoulders)), perspective
    )

    anchor_ceiling_shoulders: FloatArray2 = ceiling_shoulders[anchor, ...]
    anchor_side_shoulders: FloatArray2 = side_shoulders[anchor, ...]

    anchor_ceiling_shoulder_components: FloatArray1 = (
        anchor_ceiling_shoulders[1, :-1] - anchor_ceiling_shoulders[0, :-1]
    )
    anchor_side_shoulder_components: FloatArray1 = (
        anchor_side_shoulders[1, :-1] - anchor_side_shoulders[0, :-1]
    )

    print(f'\n{anchor_ceiling_shoulder_components = }')
    print(f'\n{anchor_side_shoulder_components = }')

    alpha, gamma = np.arccos(
        anchor_side_shoulder_components / anchor_ceiling_shoulder_components
    )
    print(f'\n{alpha = }, {gamma = }\n')

    rotation: FloatArray2 = rotation_matrix(alpha, Axis.X) @ rotation_matrix(
        gamma, Axis.Z
    )

    anchor_side_shoulders[:, 2] = 1.0

    anchor_ceiling_shoulder_components_reconstructed: FloatArray2 = (
        anchor_side_shoulders @ rotation
    )
    translation_components: FloatArray2 = (
        anchor_ceiling_shoulder_components_reconstructed
        - anchor_ceiling_shoulder_components.reshape(-1, 1)
    )

    translation: FloatArray1 = (
        translation_components[0, :] + translation_components[1, :]
    ) / 2.0
    translation = translation.reshape(-1, 1)

    return rotation, translation
