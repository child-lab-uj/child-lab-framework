import string
from math import pi

import torch
from hypothesis import assume, given, settings, strategies
from transformation_buffer.buffer import Buffer
from transformation_buffer.transformation import Transformation


def pairs_of_reference_frames(
    available_frames: list[str],
) -> strategies.SearchStrategy[tuple[str, str]]:
    return strategies.lists(
        strategies.sampled_from(available_frames),
        min_size=2,
        max_size=2,
        unique=True,
    ).map(lambda value: (value[0], value[1]))


def legal_transformations() -> strategies.SearchStrategy[Transformation]:
    translation_components = strategies.floats(
        min_value=0.0,
        max_value=10.0,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    )
    rotation_angles = strategies.floats(
        min_value=0.0,
        max_value=2 * pi,
        allow_nan=False,
        allow_infinity=False,
        allow_subnormal=False,
    )

    angles = strategies.tuples(
        rotation_angles,
        rotation_angles,
        rotation_angles,
    ).map(torch.tensor)

    translations = strategies.tuples(
        translation_components,
        translation_components,
        translation_components,
    ).map(torch.tensor)

    transformations = strategies.tuples(angles, translations).map(
        lambda components: Transformation.active(*components)
    )

    return transformations


def buffers(
    reference_frames: list[str],
    number_of_transformations: int,
) -> strategies.SearchStrategy[Buffer[str]]:
    def make_buffer(
        transformations_between_frames: list[tuple[tuple[str, str], Transformation]],
    ) -> Buffer[str]:
        buffer = Buffer[str]()

        for from_to, transformation in transformations_between_frames:
            buffer[from_to] = transformation

        return buffer

    pairs_of_frames = pairs_of_reference_frames(reference_frames)
    transformations = legal_transformations()

    buffers = strategies.lists(
        strategies.tuples(pairs_of_frames, transformations),
        min_size=number_of_transformations,
        max_size=number_of_transformations,
    ).map(make_buffer)

    return buffers


FRAMES_SMALL: list[str] = list(string.ascii_lowercase)[:10]


@given(buffers(FRAMES_SMALL, 5), pairs_of_reference_frames(FRAMES_SMALL))
@settings(max_examples=1000)
def test_bidirectional_transformation(buffer: Buffer, from_to: tuple[str, str]) -> None:
    transformation = buffer[from_to]
    inverse_transformation = buffer[(from_to[1], from_to[0])]

    # For hypothesis:
    assume(transformation is not None)
    assume(inverse_transformation is not None)

    # For mypy:
    assert transformation is not None
    assert inverse_transformation is not None

    condition = torch.isclose(
        transformation.inverse().rotation_and_translation,
        inverse_transformation.rotation_and_translation,
        atol=1e-5,
    ).all()

    assert condition
