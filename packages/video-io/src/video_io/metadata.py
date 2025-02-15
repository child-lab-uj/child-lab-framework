from typing import Self

from attrs import frozen
from torchcodec.decoders import VideoStreamMetadata  # type: ignore[attr-defined]


@frozen
class Metadata:
    fps: float
    frames: int
    width: int
    height: int

    @classmethod
    def from_stream_metadata(cls, stream_metadata: VideoStreamMetadata) -> Self:
        fps = stream_metadata.average_fps_from_header
        frames = stream_metadata.num_frames
        width = stream_metadata.width
        height = stream_metadata.height

        assert fps is not None
        assert frames is not None
        assert width is not None
        assert height is not None

        return cls(fps, frames, width, height)
