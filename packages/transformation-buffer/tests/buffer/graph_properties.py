import pytest
from icecream import ic
from transformation_buffer import Buffer, Transformation


@pytest.fixture
def buffer() -> Buffer[str]:
    return (
        Buffer[str]()
        .add_transformation('a', 'b', Transformation.identity())
        .add_transformation('a', 'c', Transformation.identity())
        .add_transformation('b', 'd', Transformation.identity())
        .add_transformation('d', 'e', Transformation.identity())
        .add_transformation('c', 'e', Transformation.identity())
        .add_transformation('d', 'f', Transformation.identity())
        .add_transformation('g', 'h', Transformation.identity())
    )


def test_frames_visible_from(buffer: Buffer[str]) -> None:
    ic(buffer.frames_of_reference)
    visible_frames = buffer.frames_visible_from('a')
    visible_frames.sort()

    expected = ['b', 'c', 'd', 'e', 'f']
    assert visible_frames == expected
