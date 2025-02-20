from math import pi
from typing import Hashable, Protocol

import torch
from attrs import frozen

from transformation_buffer.transformation import Transformation


class RigidModel[T: Hashable](Protocol):
    def transformations(self) -> dict[tuple[T, T], Transformation]: ...


@frozen
class Cube[T: Hashable]:
    width: float
    walls: tuple[T, T, T, T, T, T]

    # Cube Walls:
    #    4
    #  5 1 6
    #    2
    #    3

    def transformations(self) -> dict[tuple[T, T], Transformation]:
        first, second, third, forth, fifth, sixth = self.walls

        half_pi = pi / 2.0
        half_width = self.width / 2.0

        down = Transformation.active(
            intrinsic_euler_angles=torch.tensor((-half_pi, 0.0, 0.0)),
            translation=torch.tensor((0.0, half_width, -half_width)),
        )

        left = Transformation.active(
            intrinsic_euler_angles=torch.tensor((0.0, half_pi, 0.0)),
            translation=torch.tensor((half_width, 0.0, -half_width)),
        )

        right = Transformation.active(
            intrinsic_euler_angles=torch.tensor((0.0, -half_pi, 0.0)),
            translation=torch.tensor((-half_width, 0.0, -half_width)),
        )

        return {
            (first, second): down.clone(),
            (second, third): down.clone(),
            (third, forth): down,
            (first, fifth): left,
            (first, sixth): right,
        }
