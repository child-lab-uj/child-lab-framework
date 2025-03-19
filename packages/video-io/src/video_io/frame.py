import numpy
import torch
from jaxtyping import UInt8
from numpy.typing import NDArray

type Array = NDArray[numpy.uint8]

type ArrayRgbFrame = UInt8[numpy.ndarray, 'height width 3']
type ArrayGrayFrame = UInt8[numpy.ndarray, 'height width']

type TensorRgbFrame = UInt8[torch.Tensor, '3 height width']
