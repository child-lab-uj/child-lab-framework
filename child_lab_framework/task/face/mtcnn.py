import contextlib
import io
import typing
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

from ...core.video import Frame
from ...typing.array import (
    FloatArray1,
    IntArray1,
    IntArray2,
    IntArray3,
)

type Keypoints = dict[
    Literal[
        'right_eye',
        'left_eye',
        'nose',
        'mouth_right',
        'mouth_left',
    ],
    tuple[int, int],
]

# box has 'xywh' format
type Detection = dict[
    Literal['box', 'confidence', 'keypoints'],
    list[int] | float | Keypoints,
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

    def __init__(
        self,
        *,
        threshold: float,
        allow_upscaling: bool = True,
    ) -> None:
        self.model = MTCNN()

    def predict(self, image: Frame) -> Result | None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: ignore

        with contextlib.redirect_stdout(io.StringIO()):
            detection: list[Detection] = self.model.detect_faces(image)

        boxes: list[IntArray1] = []
        confidences: list[float] = []
        keypoints: list[IntArray2] = []

        for result in detection:
            box = typing.cast(list[int], result['box'])

            confidence = typing.cast(float, result['confidence'])

            face_keypoints = typing.cast(Keypoints, result['keypoints'])

            boxes.append(np.array(box, dtype=np.int32))
            confidences.append(confidence)

            keypoints.append(
                np.array(
                    [
                        face_keypoints['left_eye'],
                        face_keypoints['right_eye'],
                        face_keypoints['nose'],
                        face_keypoints['mouth_left'],
                        face_keypoints['mouth_right'],
                    ],
                    dtype=np.int32,
                )
            )

        if len(boxes) == 0 or len(confidences) == 0 or len(keypoints) == 0:
            return None

        return Result(
            np.stack(boxes),
            np.array(confidences, dtype=np.float32),
            np.stack(keypoints),
        )
