from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

import cv2 as opencv
import torch
from jaxtyping import UInt8

from video_io.metadata import Metadata
from video_io.visualizer import Visualizable, Visualizer


class Writer[Context: Mapping[str, object]]:
    __visualizer: Visualizer[Context]
    __encoder: opencv.VideoWriter

    def __init__(
        self,
        destination: Path,
        metadata: Metadata,
        # A funny trick to make passing the visualizer optional while keeping the type safety.
        # It's possible as long as `Visualizer` is immutable.
        visualizer: Visualizer[Context] = Visualizer(cast(Context, {})),
    ) -> None:
        if destination.exists():
            raise FileExistsError(
                f'Destination file "{destination.absolute()}" already exists'
            )

        self.__visualizer = visualizer

        self.__encoder = opencv.VideoWriter(
            str(destination),
            fourcc=self.__codec(destination.suffix),
            fps=metadata.fps,
            frameSize=(metadata.width, metadata.height),
            isColor=True,
        )

    def write(
        self,
        frame: UInt8[torch.Tensor, '3 height width'],
        annotations: Iterable[Visualizable[Context]] | None = None,
    ) -> None:
        self.write_batch(
            frame.unsqueeze(0),
            [annotations] if annotations is not None else None,
        )

    def write_batch(
        self,
        frames: UInt8[torch.Tensor, 'batch 3 height width'],
        annotations: Iterable[Iterable[Visualizable[Context]]] | None = None,
    ) -> None:
        raw_frames = frames.permute(0, 2, 3, 1).cpu().detach().numpy()

        if annotations is not None:
            self.__visualizer.annotate_batch(raw_frames, annotations)

        encoder = self.__encoder

        for frame in raw_frames:
            encoder.write(frame)

    @staticmethod
    def __codec(file_extension: str) -> int:
        match file_extension:
            case '.avi':
                return opencv.VideoWriter.fourcc(*'MJPG')

            case '.mp4':
                return opencv.VideoWriter.fourcc(*'mp4v')

            case _:
                raise UnsupportedFormatException(
                    f'File extension "{file_extension}" is not supported'
                )


class UnsupportedFormatException(Exception): ...


if __name__ == '__main__':
    from typing import Literal, TypedDict

    import numpy
    from attrs import define

    class ContextA(TypedDict):
        z: float

    @define
    class A:
        x: int

        def draw(
            self,
            frame: numpy.ndarray[tuple[int, int, Literal[3]], numpy.dtype[numpy.uint8]],
            context: ContextA,
        ) -> numpy.ndarray[tuple[int, int, Literal[3]], numpy.dtype[numpy.uint8]]:
            return frame

    # w = Writer(Path('nothing.mp4'), Metadata(0.0, 0, 0, 0))
    # w.write(torch.tensor(()), [A(10)])  # typing error

    v = Visualizer[ContextA]({'z': 10.0})
    w = Writer(Path('nothing2.mp4'), Metadata(0.0, 0, 0, 0), v)
    w.write(torch.tensor(()), [A(10)])  # ok
