import typing
import io
import numpy as np
from dataclasses import dataclass
from typing import Literal
from mtcnn.mtcnn import MTCNN
import contextlib

from ...core.video import Frame
from ...typing.array import FloatArray1, IntArray1, IntArray2, IntArray3


type Keypoints = dict[
    Literal['right_eye', 'left_eye', 'nose', 'mouth_right', 'mouth_left'],
    tuple[int, int]
]

# box has 'xywh' format
type Detection = dict[
    Literal['box', 'confidence', 'keypoints'],
    list[int] | float | Keypoints
]


@dataclass
class Result:
    boxes: IntArray2
    confidences: FloatArray1
    keypoints: IntArray3


class Detector:
    model: MTCNN
    threshold: float
    allow_upscaling: bool

    def __init__(self, *, threshold: float, allow_upscaling: bool = True) -> None:
        self.model = MTCNN()

    def predict(self, image: Frame) -> Result | None:
        with contextlib.redirect_stdout(io.StringIO()):
            detection: list[Detection] = self.model.detect_faces(image)

        boxes: list[IntArray1] = []
        confidences: list[float] = []
        keypoints: list[IntArray2] = []

        for result in detection:
            box = typing.cast(
                list[int],
                result['box']
            )

            confidence = typing.cast(
                float,
                result['confidence']
            )

            face_keypoints = typing.cast(
                Keypoints,
                result['keypoints']
            )

            boxes.append(np.array(box, dtype=np.int32))
            confidences.append(confidence)

            keypoints.append(np.array([
                face_keypoints['left_eye'],
                face_keypoints['right_eye'],
                face_keypoints['nose'],
                face_keypoints['mouth_left'],
                face_keypoints['mouth_right'],
            ], dtype=np.int32))

        if len(boxes) == 0 or len(confidences) == 0 or len(keypoints) == 0:
            return None

        return Result(
            np.stack(boxes),
            np.array(confidences, dtype=np.float32),
            np.stack(keypoints)
        )


async def main() -> None:
    from ...util import DEV_DIR
    from ...core.video import Reader, Writer, Perspective, Format
    import cv2

    reader = Reader(
        str(DEV_DIR / 'data' / 'short' / 'window_left.mp4'),
        perspective=Perspective.WINDOW_LEFT,
        batch_size=10
    )

    writer = Writer(
        str(DEV_DIR / 'output' / 'face_window_left.mp4'),
        properties=reader.properties,
        output_format=Format.MP4
    )

    model = Detector(threshold=0.5)

    reader_thread = reader.stream()
    await reader_thread.asend(None)

    writer_thread = writer.stream()
    await writer_thread.asend(None)

    while frames := await reader_thread.asend(None):
        predictions = [model.predict(frame) for frame in frames]

        annotated_frames = []

        for frame, prediction in zip(frames, predictions):
            if prediction is None:
                annotated_frames.append(frame)
                continue

            annotated_frame = frame.copy()

            box: IntArray1
            for box in prediction.boxes:
                x, y, w, h = box
                cv2.rectangle(
                    annotated_frame,
                    (x, y),
                    (x + w, y + h),
                    color=(255, 0, 0)
                )

            annotated_frames.append(annotated_frame)

        await writer_thread.asend(annotated_frames)
