from dataclasses import dataclass

from ... import pose
from ....core.video import Frame, Perspective, Properties
from ....typing.array import FloatArray1, FloatArray2
from ....typing.stream import Fiber


type Input = list[tuple[
    pose.Result | None,
    pose.Result | None,
    FloatArray2,
    FloatArray2
]]


@dataclass
class Result:
    translation: FloatArray1
    rotation: FloatArray2


class Estimator:
    from_view: Properties
    to_view: Properties

    def __init__(self, from_view: Properties, to_view: Properties) -> None:
        self.from_view = from_view
        self.to_view = to_view

    def predict(
        self,
        from_pose: pose.Result,
        to_pose: pose.Result,
        from_depth: FloatArray2,
        to_depth: FloatArray2
    ) -> Result:
        ...

    async def stream(self) -> Fiber[Input | None, list[Result | None] | None]:
        results: list[Result | None] | None = None

        while True:
            match (yield results):
                case list(inputs):
                    ...

                case _:
                    results = None
