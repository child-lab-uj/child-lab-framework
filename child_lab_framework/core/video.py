from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from typing import Literal
import typing
from cv2.typing import MatLike
import numpy as np
import cv2

from ..typing.stream import Fiber
from ..typing.video import Frame
from ..typing.array import IntArray2, FloatArray2
from .stream import autostart, nones


class Format(Enum):
    AVI = "mjpg"
    MP4 = "mp4v"


class Perspective(IntEnum):
    CEILING = auto()
    WINDOW_RIGHT = auto()
    WINDOW_LEFT = auto()
    WALL_RIGHT = auto()
    WALL_LEFT = auto()


@dataclass
class Properties:
    width: int
    height: int
    fps: int
    perspective: Perspective


class Reader:
    source: str
    batch_size: int
    decoder: cv2.VideoCapture
    properties: Properties

    def __init__(
        self, source: str, *, perspective: Perspective, batch_size: int
    ) -> None:
        self.source = source
        self.batch_size = batch_size

        decoder = cv2.VideoCapture(source)
        self.decoder = decoder

        width = int(decoder.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(decoder.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(decoder.get(cv2.CAP_PROP_FPS))

        self.properties = Properties(width, height, fps, perspective)

    def __del__(self) -> None:
        self.decoder.release()

    def stream(self) -> Fiber[None, list[Frame] | None]:
        decoder = self.decoder
        batch_size = self.batch_size

        while decoder.isOpened():
            batch: list[Frame] | None = []

            success = False

            for _ in range(batch_size):
                success, frame = decoder.read()

                if not success:
                    break

                frame.flags.writeable = False
                batch.append(typing.cast(Frame, frame))

            if not success or len(batch) == 0:
                break

            yield batch

        yield from nones()


class Writer:
    destination: str
    properties: Properties
    output_format: Format
    encoder: cv2.VideoWriter

    def __init__(
        self, destination: str, properties: Properties, *, output_format: Format
    ) -> None:
        self.destination = destination
        self.properties = properties
        self.output_format = output_format

        self.encoder = cv2.VideoWriter(
            destination,
            fourcc=cv2.VideoWriter.fourcc(*output_format.value),
            fps=int(properties.fps),
            frameSize=(int(properties.width), int(properties.height)),
            isColor=True,
        )

    def __del__(self) -> None:
        self.encoder.release()

    @autostart
    def stream(self) -> Fiber[list[Frame] | None, None]:
        while True:
            match (yield):
                case list(frames):
                    for frame in frames:
                        self.encoder.write(frame)

                case None:
                    continue


def cropped(frame: Frame, boxes: FloatArray2) -> list[Frame]:
    boxes_truncated: IntArray2 = boxes.astype(np.int32)

    crops = [
        frame[box[0]:box[2], box[1]:box[3]]
        for box in list(boxes_truncated)
    ]

    return crops
