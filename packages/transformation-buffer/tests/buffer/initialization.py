from transformation_buffer import Buffer


def test_init_with_duplicated_frames() -> None:
    frames_of_reference = ['frame1', 'frame1', 'frame2', 'frame3']
    buffer = Buffer(frames_of_reference)

    assert buffer.frames_of_reference == frozenset(frames_of_reference)
