import asyncio
import math
import typing
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from functools import lru_cache

import cv2
import numpy as np

from ..typing.array import FloatArray1, FloatArray2, IntArray2
from ..typing.stream import Fiber
from ..typing.video import Frame
from .stream import autostart


class Format(Enum):
    AVI = 'mjpg'
    MP4 = 'mp4v'


class Perspective(IntEnum):
    CEILING = auto()
    WINDOW_RIGHT = auto()
    WINDOW_LEFT = auto()
    WALL_RIGHT = auto()
    WALL_LEFT = auto()


@dataclass(unsafe_hash=True, frozen=True)
class Calibration:
    optical_center: tuple[float, float]
    focal_length: tuple[float, float]

    @staticmethod
    def heuristic(width: int, height: int) -> 'Calibration':
        cx = width / 2.0
        cy = height / 2.0

        fx = 500.0 * width / 640.0
        fy = 500.0 * height / 480.0
        fx = (fx + fy) / 2.0
        fy = fx

        return Calibration((cx, cy), (fx, fy))

    @lru_cache(1)
    def flat(self) -> tuple[float, float, float, float]:
        return (*self.focal_length, *self.optical_center)

    @lru_cache(1)
    def intrinsics(self) -> FloatArray2:
        m = np.zeros((3, 4), dtype=np.float32)
        m[0, 0], m[1, 1] = self.focal_length
        m[0:2, 2] = self.optical_center
        m[2, 2] = 1.0

        return m

    @lru_cache(1)
    def distortion(self) -> FloatArray1:
        return np.zeros(4, dtype=np.float32)


class Properties:
    width: int
    height: int
    fps: int
    perspective: Perspective
    calibration: Calibration

    def __init__(
        self, width: int, height: int, fps: int, perspective: Perspective
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.perspective = perspective
        self.calibration = Calibration.heuristic(width, height)

    def __repr__(self) -> str:
        width = self.width
        height = self.height
        fps = self.fps
        perspective = self.perspective
        calibration = self.calibration

        return f'Properties:\n{width = },\n{height = },\n{fps = },\n{perspective = },\n{calibration = }'


class Reader:
    source: str
    batch_size: int
    decoder: cv2.VideoCapture

    __input_properties: Properties
    __mimicked_properties: Properties
    properties: Properties

    def __init__(
        self,
        source: str,
        *,
        perspective: Perspective,
        batch_size: int,
        like: Properties | None = None,
    ) -> None:
        self.source = source
        self.batch_size = batch_size

        decoder = cv2.VideoCapture(source)
        self.decoder = decoder

        width = int(decoder.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(decoder.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(decoder.get(cv2.CAP_PROP_FPS))

        input_properties = Properties(width, height, fps, perspective)
        self.__input_properties = input_properties
        self.__mimicked_properties = self.properties = like or input_properties

    def __del__(self) -> None:
        self.decoder.release()

    async def stream(self) -> Fiber[None, list[Frame] | None]:
        decoder = self.decoder
        batch_size = self.batch_size

        input_properties = self.__input_properties
        mimicked_properties = self.__mimicked_properties
        destination_size = mimicked_properties.width, mimicked_properties.height

        frame_repeats = int(math.ceil(mimicked_properties.fps / input_properties.fps))

        fx, fy = input_properties.calibration.focal_length

        while decoder.isOpened():
            batch: list[Frame] | None = []

            success = False

            for _ in range(batch_size):
                success, frame = decoder.read()

                if not success:
                    break

                resized_frame = cv2.resize(frame, destination_size, fx=fx, fy=fy)
                resized_frame.flags.writeable = False

                for _ in range(frame_repeats):
                    batch.append(typing.cast(Frame, resized_frame.copy()))

            if not success or len(batch) == 0:
                break

            yield batch
            await asyncio.sleep(0.0)

        while True:
            yield None


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
    async def stream(self) -> Fiber[list[Frame] | None, None]:
        while True:
            match (yield):
                case list(frames):
                    for frame in frames:
                        self.encoder.write(frame)

                case None:
                    continue

            await asyncio.sleep(0.0)


def cropped(frame: Frame, boxes: FloatArray2) -> list[Frame]:
    boxes_truncated: IntArray2 = boxes.astype(np.int32)
    crops = [frame[box[0] : box[2], box[1] : box[3]] for box in list(boxes_truncated)]

    return crops
