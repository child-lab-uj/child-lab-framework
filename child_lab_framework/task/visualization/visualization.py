import typing
from collections.abc import Sequence
from functools import reduce
from itertools import repeat, starmap

from ...core.video import Frame, Properties
from ...typing.stream import Fiber
from .configuration import Configuration


class Visualizable[T: Configuration](typing.Protocol):
    def visualize(
        self,
        frame: Frame,
        frame_properties: Properties,
        configuration: T,
    ) -> Frame: ...


class Visualizer[T: Configuration]:
    properties: Properties
    configuration: T

    def __init__(
        self,
        *,
        properties: Properties,
        configuration: T,
    ) -> None:
        self.properties = properties
        self.configuration = configuration

    def annotate(self, frame: Frame, *results: Visualizable[T]) -> Frame:
        return self.__annotate(frame, *results)

    def annotate_batch(
        self,
        frames: list[Frame],
        *results: Sequence[Visualizable[T] | None] | None,
    ) -> list[Frame]:
        return list(
            starmap(
                self.__annotate,
                zip(frames, *filter(None, results)),
            )
        )

    def __annotate(self, frame: Frame, *results: Visualizable[T] | None) -> Frame:
        frame = frame.copy()
        frame.flags.writeable = True
        properties = self.properties
        configuration = self.configuration

        return reduce(
            lambda frame, result: result.visualize(frame, properties, configuration),
            filter(None, results),
            frame,
        )

    async def stream(
        self,
    ) -> Fiber[
        tuple[list[Frame] | None, *tuple[list[Visualizable[T] | None] | None, ...]],
        list[Frame] | None,
    ]:
        annotated_frames: list[Frame] | None = None

        while True:
            match (yield annotated_frames):
                case list(frames), *results if any(results):
                    annotated_frames = list(
                        starmap(
                            self.__annotate,
                            zip(
                                frames,
                                *map(
                                    lambda result: result or repeat(None),
                                    results,
                                ),
                            ),
                        )
                    )

                case list(frames), *_:
                    annotated_frames = frames

                case _:
                    annotated_frames = None
