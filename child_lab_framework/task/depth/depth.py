import typing
from functools import lru_cache
from typing import Literal
import numpy as np
import onnxruntime as onnx
import cv2

from ...core.video import Frame
from ...core import image
from ...util import MODELS_DIR
from ...typing.array import FloatArray1, FloatArray2, FloatArray3, FloatArray4


type OnnxInputDict = dict[
    Literal['pixel_values'],
    FloatArray4
]

type PaddingInfo = tuple[
    tuple[int, int],
    tuple[int, int]
]


@lru_cache(2, typed=True)
def padding_info(height: int, width: int, expected_height: int, expected_width: int) -> PaddingInfo:
    y_padding = expected_height - height
    x_padding = expected_width - width

    y_padding_half = y_padding // 2
    x_padding_half = x_padding // 2

    return (
        (y_padding_half, y_padding - y_padding_half),
        (x_padding_half, x_padding - x_padding_half)
    )


def resized(
    frame: Frame,
    size: tuple[int, int]
) -> Frame:
    height, width, _ = frame.shape
    expected_height, expected_width = size
    scale = min(expected_height / height, expected_width / width)

    resized_width = int(width * scale)
    resized_height = int(height * scale)
    resized_frame = image.resized(frame, resized_height, resized_width)

    return resized_frame


def padded(
    frame: FloatArray3,
    expected_size: tuple[int, int],
    color: tuple[float, float, float]
) -> tuple[FloatArray3, PaddingInfo]:
    padding = padding_info(*frame.shape[:2], *expected_size)

    padded_frame = typing.cast(
        FloatArray3,
        cv2.copyMakeBorder(
            frame,
            *padding[0],
            *padding[1],
            cv2.BORDER_CONSTANT,
            value=color
        )
    )

    return padded_frame, padding


class Estimator:
    EXECUTION_PROVIDER = 'CPUExecutionProvider'
    MODEL_PATH = str(MODELS_DIR / 'metric3d-vit-small.onnx')
    MODEL_INPUT_SIZE = (616, 1064)
    PADDING_BORDER_COLOR = (123.675, 116.28, 103.53)

    session: onnx.InferenceSession

    def __init__(self) -> None:
        self.session = onnx.InferenceSession(
            self.MODEL_PATH,
            providers=[self.EXECUTION_PROVIDER]
        )

    def __prepare_input(self, frame: Frame) -> tuple[OnnxInputDict, PaddingInfo]:
        resized_frame = resized(frame, self.MODEL_INPUT_SIZE)

        padded_frame, padding = padded(
            resized_frame.astype(np.float32),
            self.MODEL_INPUT_SIZE,
            self.PADDING_BORDER_COLOR
        )

        # 1, 3, H, W
        packed_frame: FloatArray4 = np.ascontiguousarray(
            np.transpose(padded_frame, (2, 0, 1))[np.newaxis],
            dtype=np.float32
        )

        onnx_input: OnnxInputDict = {'pixel_values': packed_frame}

        return onnx_input, padding

    def __call__(self, frame: Frame) -> tuple[FloatArray2, FloatArray3]:
        return self.predict(frame)

    def predict(self, frame: Frame) -> tuple[FloatArray2, FloatArray3]:
        onnx_input, padding_info = self.__prepare_input(frame)
        results = self.session.run(None, onnx_input)

        height, width, _ = frame.shape
        result_rows = slice(padding_info[0][0], height - padding_info[0][1])
        result_columns = slice(padding_info[1][0], width - padding_info[1][1])

        depth: FloatArray2 = results[0].squeeze()[result_rows, result_columns]
        depth = image.resized(depth, height, width)

        normals: FloatArray3 = np.transpose(
            results[1].squeeze(),
            (1, 2, 0)
        )[result_rows, result_columns, :]
        normals = image.resized(normals, height, width)

        return depth, normals


def main() -> None:
    import sys
    from ...util import DEV_DIR
    from ...core.video import Reader, Writer, Perspective, Format

    def to_frame(depth_map: FloatArray2) -> Frame:
        green = (depth_map / depth_map.max() * 255).astype(np.uint8)
        other = np.zeros_like(green)
        frame = np.transpose(np.stack((other, green, other)), (1, 2, 0))
        return frame


    reader = Reader(
        str(DEV_DIR / 'data/short/window_left.mp4'),
        perspective=Perspective.CEILING,
        batch_size=10
    )

    writer = Writer(
        str(DEV_DIR / 'output/depth.mp4'),
        reader.properties,
        output_format=Format.MP4
    )

    reader_thread = reader.stream()
    writer_thread = writer.stream()

    estimator = Estimator()

    frames_count = 0
    fps = reader.properties.fps

    while frames := reader_thread.send(None):
        writer_thread.send([to_frame(estimator.predict(frame)[0]) for frame in frames])

        frames_count += len(frames)
        seconds = frames_count / fps

        sys.stdout.write(f'\r{seconds = :.2f}')
        sys.stdout.flush()
