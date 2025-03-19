from collections.abc import Generator
from pathlib import Path
from typing import Annotated, Final, cast

import torch
from annotated_types import Gt
from jaxtyping import UInt8
from more_itertools import take
from torchcodec.decoders import (  # type: ignore[attr-defined]
    VideoDecoder,
    VideoStreamMetadata,
)
from torchvision.transforms import Compose, Resize

from . import Metadata


class Reader:
    __decoder: VideoDecoder
    __transformation: Compose
    __frame_indices: Generator[int, None, None]

    metadata: Final[Metadata]
    device: Final[torch.device]

    def __init__(
        self,
        source: Path,
        device: torch.device = torch.device('cpu'),
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.device = device

        self.__decoder = VideoDecoder(
            source,
            device='cpu' if device == torch.device('mps') else str(device),
        )

        self.metadata = metadata = Metadata.from_stream_metadata(
            # `VideoDecoder.metadata` has strange typing which forces a manual downcast ;v
            cast(VideoStreamMetadata, self.__decoder.metadata)
        )

        if fps is None:
            self.__frame_indices = (i for i in range(metadata.frames))
        else:
            frames = metadata.frames
            interpolated_length = int(frames * fps / metadata.fps)

            self.__frame_indices = (
                int(i)
                for i in torch.linspace(
                    0,
                    frames,
                    steps=interpolated_length,
                    dtype=torch.int,
                )
            )

        match height, width:
            case None, None:
                self.__transformation = Compose(())  # type: ignore[no-untyped-call]

            case None, int(w):
                self.__transformation = Compose([Resize((metadata.height, w))])  # type: ignore[no-untyped-call]

            case int(h), None:
                self.__transformation = Compose([Resize((h, metadata.width))])  # type: ignore[no-untyped-call]

            case int(h), int(w):
                self.__transformation = Compose([Resize((h, w))])  # type: ignore[no-untyped-call]

    def read(self) -> UInt8[torch.Tensor, '3 height width'] | None:
        match next(self.__frame_indices, None):
            case None:
                return None

            case index:
                return self.__decoder.get_frame_at(index).data

    def read_batch(
        self,
        size: Annotated[int, Gt(0)],
    ) -> UInt8[torch.Tensor, 'size 3 height width'] | None:
        assert size > 0, 'Expected positive batch size'

        indices = take(size, self.__frame_indices)
        if len(indices) == 0:
            return None

        frames = self.__decoder.get_frames_at(indices).data.to(self.device)
        return cast(torch.Tensor, self.__transformation(frames))
