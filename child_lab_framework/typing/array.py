from typing import Literal
import numpy

type FloatArray1 = numpy.ndarray[Literal[1], numpy.dtype[numpy.float32]]
type FloatArray2 = numpy.ndarray[Literal[2], numpy.dtype[numpy.float32]]
type FloatArray3 = numpy.ndarray[Literal[3], numpy.dtype[numpy.float32]]

type ByteArray2 = numpy.ndarray[Literal[2], numpy.dtype[numpy.uint8]]
type ByteArray3 = numpy.ndarray[Literal[3], numpy.dtype[numpy.uint8]]

type IntArray2 = numpy.ndarray[Literal[2], numpy.dtype[numpy.int32]]
type IntArray3 = numpy.ndarray[Literal[3], numpy.dtype[numpy.int32]]
