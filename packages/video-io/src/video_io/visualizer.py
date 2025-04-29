from collections.abc import Iterable, Mapping
from typing import Any, Literal, Protocol, TypedDict

import numpy
from attrs import frozen

type RgbFrame = numpy.ndarray[tuple[int, int, Literal[3]], numpy.dtype[numpy.uint8]]


class Visualizable[Context: Mapping[str, Any]](Protocol):
    def draw(
        self,
        frame: RgbFrame,
        context: Context,
    ) -> RgbFrame: ...


@frozen
class Visualizer[Context: Mapping[str, Any]]:
    context: Context

    def annotate(
        self,
        frame: RgbFrame,
        items: Iterable[Visualizable[Context]],
    ) -> RgbFrame:
        context = self.context

        for item in items:
            item.draw(frame, context)

        return frame

    def annotate_batch(
        self,
        frames: Iterable[RgbFrame],
        items: Iterable[Iterable[Visualizable[Context]]],
    ) -> list[RgbFrame]:
        return [self.annotate(frame, items) for frame, items in zip(frames, items)]


# A simple example on how to use 'contexts' in a type-safe way.
if __name__ == '__main__':

    class SampleContext(TypedDict):
        x: int
        y: float

    class ExtendedContext(TypedDict):
        x: int
        y: float
        z: str

    @frozen
    class A:
        x: int

        def draw(self, frame: RgbFrame, context: SampleContext) -> RgbFrame:
            return frame

    vis = Visualizer[ExtendedContext]({'x': 1, 'y': 1.0, 'z': ''})

    vis.annotate(numpy.array(()), [A(10)])
