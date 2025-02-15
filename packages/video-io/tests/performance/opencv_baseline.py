from pathlib import Path
from typing import Literal

import cv2 as opencv
import numpy


class PoorMansReader:
    __decoder: opencv.VideoCapture
    __height: int
    __width: int

    def __init__(self, source: Path) -> None:
        assert source.is_file()

        self.__decoder = decoder = opencv.VideoCapture(str(source))
        self.__height = int(decoder.get(opencv.CAP_PROP_FRAME_HEIGHT))
        self.__width = int(decoder.get(opencv.CAP_PROP_FRAME_WIDTH))

    def read_batch(
        self,
        size: int,
    ) -> numpy.ndarray[tuple[int, int, int, Literal[3]], numpy.dtype[numpy.uint8]] | None:
        batch = numpy.empty((size, self.__height, self.__width, 3), dtype=numpy.uint8)

        decoder = self.__decoder

        for i in range(size):
            success, _ = decoder.read(batch[i, ...])
            if not success:
                return None

        return batch
