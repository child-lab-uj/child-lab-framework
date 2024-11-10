import asyncio
import math
import typing
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2

from ..typing.stream import Fiber
from ..typing.video import Frame
from .calibration import Calibration


class Format(Enum):
    AVI = 'mjpg'
    MP4 = 'mp4v'


@dataclass(frozen=True, repr=False)
class Properties:
    length: int
    height: int
    width: int
    fps: int
    calibration: Calibration

    def __repr__(self) -> str:
        width = self.width
        height = self.height
        fps = self.fps
        calibration = self.calibration

        return f'Properties:\n{width = },\n{height = },\n{fps = },\n{calibration = }'


@dataclass(frozen=True)
class Input:
    name: str
    source: Path
    calibration: Calibration | None = None

    def __post_init__(self) -> None:
        if not self.source.is_file():
            raise ValueError(f'Invalid video source: {self.source}')


class Reader:
    source: Path
    batch_size: int
    decoder: cv2.VideoCapture

    properties: Properties
    __input_properties: Properties

    __frame_repetitions: int
    __read_repetitions_left: int
    __read_last_frame: Frame

    def __init__(
        self,
        input: Input,
        *,
        batch_size: int,
        height: int | None = None,
        width: int | None = None,
        fps: int | None = None,
    ) -> None:
        self.input = input
        self.batch_size = batch_size

        self.decoder = decoder = cv2.VideoCapture(str(input.source))
        input_length = int(decoder.get(cv2.CAP_PROP_FRAME_COUNT))
        input_height = int(decoder.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_width = int(decoder.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_fps = int(decoder.get(cv2.CAP_PROP_FPS))

        mimicked_height = height or input_height
        mimicked_width = width or input_width
        mimicked_fps = fps or input_fps

        self.__frame_repetitions = 1 if fps is None else int(math.ceil(fps / input_fps))
        self.__read_repetitions_left = 0
        self.__read_last_frame = None  # type: ignore  # A kind of `lateinit` field

        input_calibration = input.calibration or Calibration.heuristic(
            input_height, input_width
        )

        output_calibration = input_calibration.resized(
            1.0 if width is None else width / input_width,
            1.0 if height is None else height / input_height,
        )

        self.__input_properties = Properties(
            input_length,
            input_height,
            input_width,
            input_fps,
            input_calibration,
        )

        # Output properties with maybe mimicked parameters
        self.properties = Properties(
            input_length * self.__frame_repetitions,
            mimicked_height,
            mimicked_width,
            mimicked_fps,
            output_calibration,
        )

    def __del__(self) -> None:
        self.decoder.release()

    # TODO: wrap frames in dataclasses and add flag signalizing whether the frame was imputed.
    # Ignore imputed frames in `Reader` and squash their annotations to their "parent" frames.
    def read(self) -> Frame | None:
        decoder = self.decoder

        if not decoder.isOpened():
            return None

        if self.__read_repetitions_left > 0:
            self.__read_repetitions_left -= 1
            return self.__read_last_frame

        frame: Frame
        success, frame = decoder.read()  # type: ignore

        if not success:
            return None

        fx, fy = self.__input_properties.calibration.focal_length

        output_properties = self.properties
        output_size = output_properties.width, output_properties.height

        frame = cv2.resize(frame, output_size, fx=fx, fy=fy)  # type: ignore
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore
        frame.flags.writeable = False

        self.__read_repetitions_left = self.__frame_repetitions - 1
        self.__read_last_frame = frame

        return frame

    def read_skipping(self, skip: int) -> Frame | None:
        for _ in range(skip - 1):
            self.read()

        return self.read()

    def read_batch(self) -> list[Frame] | None:
        decoder = self.decoder
        repetitions = self.__frame_repetitions

        if not decoder.isOpened():
            return None

        fx, fy = self.__input_properties.calibration.focal_length

        output_properties = self.properties
        output_size = output_properties.width, output_properties.height

        frame: Frame
        batch: list[Frame] = []

        for _ in range(self.batch_size):
            success, frame = decoder.read()  # type: ignore

            if not decoder.isOpened():
                break

            if not success:
                return None

            frame = cv2.resize(frame, output_size, fx=fx, fy=fy)  # type: ignore
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore
            frame.flags.writeable = False

            for _ in range(repetitions):
                batch.append(typing.cast(Frame, frame.copy()))

        return batch

    async def stream(self) -> Fiber[None, list[Frame] | None]:
        decoder = self.decoder
        batch_size = self.batch_size

        input_properties = self.__input_properties
        output_properties = self.properties
        destination_size = output_properties.width, output_properties.height

        frame_repeats = int(math.ceil(output_properties.fps / input_properties.fps))

        fx, fy = input_properties.calibration.focal_length

        yield None  # workaround for codegen adding `await asend(None)` upon opening each stream

        while decoder.isOpened():
            batch: list[Frame] | None = []

            success = False

            for _ in range(batch_size):
                success, frame = decoder.read()

                if not success:
                    break

                frame = cv2.resize(frame, destination_size, fx=fx, fy=fy)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame.flags.writeable = False

                for _ in range(frame_repeats):
                    batch.append(typing.cast(Frame, frame.copy()))

            if not success or len(batch) == 0:
                break

            yield batch
            await asyncio.sleep(0.0)

        while True:
            yield None


class Writer:
    destination: Path
    properties: Properties
    output_format: Format
    encoder: cv2.VideoWriter

    def __init__(
        self,
        destination: Path,
        properties: Properties,
        *,
        output_format: Format,
    ) -> None:
        self.destination = destination
        self.properties = properties
        self.output_format = output_format

        # TODO: Implement general resolution (Issue #61)
        destination_with_resolved_format = (
            str(destination).removesuffix(destination.suffix) + '.mp4'
            if output_format == Format.MP4
            else '.avi'
        )

        self.encoder = cv2.VideoWriter(
            destination_with_resolved_format,
            fourcc=cv2.VideoWriter.fourcc(*output_format.value),
            fps=int(properties.fps),
            frameSize=(int(properties.width), int(properties.height)),
            isColor=True,
        )

    def __del__(self) -> None:
        self.encoder.release()

    def write(self, frame: Frame) -> None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # type: ignore
        self.encoder.write(frame)

    def write_batch(self, frames: list[Frame]) -> None:
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # type: ignore
            self.encoder.write(frame)

    async def stream(self) -> Fiber[list[Frame] | None, None]:
        while True:
            match (yield):
                case list(frames):
                    self.write_batch(frames)

                case None:
                    continue

            await asyncio.sleep(0.0)
