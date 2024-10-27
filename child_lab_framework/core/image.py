import typing

import cv2
import numpy as np

from ..typing.array import FloatArray2, IntArray2
from ..typing.video import Frame

Input = typing.TypeVar('Input', covariant=True, bound=np.ndarray)


# NOTE: introduce an enum with interpolation variants if needed
def resized[Input](frame: Input, height: int, width: int) -> Input:
    return typing.cast(
        Input,
        cv2.resize(
            typing.cast(cv2.typing.MatLike, frame),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        ),
    )


def cropped(frame: Frame, boxes: FloatArray2) -> list[Frame]:
    boxes_truncated: IntArray2 = boxes.astype(np.int32)

    crops = [frame[box[0] : box[2], box[1] : box[3]] for box in list(boxes_truncated)]

    return crops
