import numpy

type FloatArray1 = numpy.ndarray[tuple[int], numpy.dtype[numpy.float32]]
type FloatArray2 = numpy.ndarray[tuple[int, int], numpy.dtype[numpy.float32]]
type FloatArray3 = numpy.ndarray[tuple[int, int, int], numpy.dtype[numpy.float32]]
type FloatArray4 = numpy.ndarray[tuple[int, int, int, int], numpy.dtype[numpy.float32]]

type FloatArray6 = numpy.ndarray[
    tuple[int, int, int, int, int, int], numpy.dtype[numpy.float32]
]

type PointCloud = FloatArray6

type ByteArray2 = numpy.ndarray[tuple[int, int], numpy.dtype[numpy.uint8]]
type ByteArray3 = numpy.ndarray[tuple[int, int, int], numpy.dtype[numpy.uint8]]

type IntArray1 = numpy.ndarray[tuple[int], numpy.dtype[numpy.int32]]
type IntArray2 = numpy.ndarray[tuple[int, int], numpy.dtype[numpy.int32]]
type IntArray3 = numpy.ndarray[tuple[int, int, int], numpy.dtype[numpy.int32]]

type BoolArray1 = numpy.ndarray[tuple[int], numpy.dtype[numpy.bool_]]
type BoolArray2 = numpy.ndarray[tuple[int, int], numpy.dtype[numpy.bool_]]
type BoolArray3 = numpy.ndarray[tuple[int, int, int], numpy.dtype[numpy.bool_]]
