import asyncio
import math
import typing
from copy import copy
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import yaml

from ..typing.array import FloatArray1, FloatArray2, FloatArray3
from ..typing.stream import Fiber
from ..typing.video import Frame


class Format(Enum):
    AVI = 'mjpg'
    MP4 = 'mp4v'


class Perspective(Enum):
    CEILING = 'Ceiling'
    WINDOW_RIGHT = 'Window Right'
    WINDOW_LEFT = 'Window Left'
    WALL_RIGHT = 'Wall Right'
    WALL_LEFT = 'Wall Left'
    OTHER = 'Other'


@dataclass(unsafe_hash=True, frozen=True)
class Calibration:
    optical_center: tuple[float, float]
    focal_length: tuple[float, float]
    distortion: FloatArray1

    @staticmethod
    def heuristic(width: int, height: int) -> 'Calibration':
        cx = width / 2.0
        cy = height / 2.0

        fx = 500.0 * width / 640.0
        fy = 500.0 * height / 480.0
        fx = (fx + fy) / 2.0
        fy = fx

        distortion = np.zeros(5, dtype=np.float32)

        return Calibration((cx, cy), (fx, fy), distortion)

    # @lru_cache(1)
    def flat(self) -> tuple[float, float, float, float]:
        return (*self.focal_length, *self.optical_center)

    @property
    def intrinsics(self) -> FloatArray2:
        m = np.zeros((3, 3), dtype=np.float32)
        m[0, 0], m[1, 1] = self.focal_length
        m[0:2, 2] = self.optical_center
        m[2, 2] = 1.0

        return m

    def depth_to_3D(self, depth: FloatArray2) -> FloatArray3:
        # TODO: remove distortion using opencv and distortion array, check if the shape is okay
        u = np.arange(depth.shape[1])
        v = np.arange(depth.shape[0])
        u, v = np.meshgrid(u, v)

        x = (u - self.optical_center[0]) * depth / self.focal_length[0]
        y = (v - self.optical_center[1]) * depth / self.focal_length[1]
        z = depth

        # Expand dimensions to make it 3D
        x = np.expand_dims(x, axis=-1)
        y = np.expand_dims(y, axis=-1)
        z = np.expand_dims(z, axis=-1)

        return np.concatenate((x, y, z), axis=-1)

    def to_yaml(self, file_path: str) -> None:
        data = {
            'optical_center': self.optical_center,
            'focal_length': self.focal_length,
            'distortion': self.distortion.tolist(),
        }
        with open(file_path, 'w+') as file:
            yaml.dump(data, file)

    @staticmethod
    def from_yaml(file_path: str) -> 'Calibration':
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            optical_center = tuple(data['optical_center'])
            focal_length = tuple(data['focal_length'])
            distortion = np.array(data['distortion'], dtype=np.float32)
            return Calibration(optical_center, focal_length, distortion)


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

    __frame_repetitions: int

    __read_repetitions_left: int
    __read_last_frame: Frame

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

        self.__mimicked_properties = self.properties = copy(like) or input_properties
        self.properties.perspective = perspective

        self.__frame_repetitions = int(
            math.ceil(self.__mimicked_properties.fps / input_properties.fps)
        )

        self.__read_repetitions_left = 0
        self.__read_last_frame = None  # type: ignore  # A kind of `lateinit` field

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

        mimicked_properties = self.__mimicked_properties
        destination_size = mimicked_properties.width, mimicked_properties.height

        frame = cv2.resize(frame, destination_size, fx=fx, fy=fy)  # type: ignore
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore
        frame.flags.writeable = False

        self.__read_repetitions_left = self.__frame_repetitions - 1
        self.__read_last_frame = frame

        return frame

    def read_batch(self) -> list[Frame] | None:
        decoder = self.decoder
        repetitions = self.__frame_repetitions

        if not decoder.isOpened():
            return None

        fx, fy = self.__input_properties.calibration.focal_length

        mimicked_properties = self.__mimicked_properties
        destination_size = mimicked_properties.width, mimicked_properties.height

        frame: Frame
        batch: list[Frame] = []

        for _ in range(self.batch_size):
            success, frame = decoder.read()  # type: ignore

            if not decoder.isOpened():
                break

            if not success:
                return None

            frame = cv2.resize(frame, destination_size, fx=fx, fy=fy)  # type: ignore
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore
            frame.flags.writeable = False

            for _ in range(repetitions):
                batch.append(typing.cast(Frame, frame.copy()))

        return batch

    async def stream(self) -> Fiber[None, list[Frame] | None]:
        decoder = self.decoder
        batch_size = self.batch_size

        input_properties = self.__input_properties
        mimicked_properties = self.__mimicked_properties
        destination_size = mimicked_properties.width, mimicked_properties.height

        frame_repeats = int(math.ceil(mimicked_properties.fps / input_properties.fps))

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
