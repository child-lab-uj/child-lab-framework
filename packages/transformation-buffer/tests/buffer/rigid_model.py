from itertools import product
from math import pi

import pytest
import torch
from transformation_buffer.buffer import Buffer
from transformation_buffer.rigid_model import Cube
from transformation_buffer.transformation import Transformation

WALLS = 'wall1', 'wall2', 'wall3', 'wall4', 'wall5', 'wall6'


@pytest.fixture
def buffer() -> Buffer[str]:
    return Buffer[str]().add_object(Cube(1.0, WALLS))


def test_all_walls_reachable(buffer: Buffer[str]) -> None:
    for from_wall, to_wall in product(WALLS, WALLS):
        assert buffer[from_wall, to_wall] is not None, (
            f'Transformation from {from_wall} to {to_wall} is not present in the buffer.'
        )


def test_wall2_to_wall6(buffer: Buffer[str]) -> None:
    two_to_six = buffer['wall2', 'wall6']

    expected = Transformation.active(
        intrinsic_euler_angles=torch.tensor((pi / 2.0, 0.0, pi / 2.0)),
        translation=torch.tensor((0.0, -0.5, -0.5)),
    )

    assert two_to_six is not None
    assert Transformation.approx_eq(two_to_six, expected, absolute_tolerance=1e-7)


def test_wall4_to_wall6(buffer: Buffer[str]) -> None:
    four_to_six = buffer['wall4', 'wall6']

    expected = Transformation.active(
        intrinsic_euler_angles=torch.tensor((-pi / 2.0, 0.0, -pi / 2.0)),
        translation=torch.tensor((0.0, 0.5, -0.5)),
    )

    assert four_to_six is not None
    assert Transformation.approx_eq(four_to_six, expected, absolute_tolerance=1e-6)


def test_wall4_to_wall5(buffer: Buffer[str]) -> None:
    four_to_five = buffer['wall4', 'wall5']

    expected = Transformation.active(
        intrinsic_euler_angles=torch.tensor((-pi / 2.0, 0.0, pi / 2.0)),
        translation=torch.tensor((0.0, 0.5, -0.5)),
    )

    assert four_to_five is not None
    assert Transformation.approx_eq(four_to_five, expected, absolute_tolerance=1e-6)
