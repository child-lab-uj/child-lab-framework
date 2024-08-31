import asyncio
import os
from pathlib import Path
from filelock import BaseFileLock, FileLock
from typing import Literal, TextIO
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from scipy.spatial.distance import pdist
import numpy as np
import numba

from .. import pose
from ...typing.array import FloatArray2
from ...typing.stream import Fiber


@dataclass
class Result:
    actors: list[pose.Actor]
    distances: FloatArray2
    # TODO: additional results (e.g. related to time series)


class Estimator:
    executor: ThreadPoolExecutor

    def __init__(self, executor: ThreadPoolExecutor) -> None:
        self.executor = executor

    def predict(self, poses: pose.Result) -> Result:
        centres = poses.centres
        distances = pdist(centres.T)
        return Result(poses.actors, distances)

    async def stream(self) -> Fiber[list[pose.Result | None] | None, list[Result | None] | None]:
        loop = asyncio.get_running_loop()
        executor = self.executor

        results: list[Result | None] | None = None

        while True:
            match (yield results):
                case list(poses):
                    results = await loop.run_in_executor(
                        executor,
                        lambda: [
                            self.predict(pose)
                            if pose is not None
                            else None
                            for pose in poses
                        ]
                    )

                case _:
                    results = None


class FileLogger:
    DELIMITER = ';'

    destination_location: Path
    destination: TextIO

    def __init__(self, destination: str) -> None:
        self.destination_location = Path(destination)
        self.destination = open(destination, 'w+')

    def __del__(self) -> None:
        self.destination.close()

    async def stream(self) -> Fiber[list[Result | None] | None, None]:
        DELIMITER = self.DELIMITER

        destination = self.destination
        destination.write(f'frame{DELIMITER}distance\n')

        frame = 0

        while True:
            match (yield):
                case list(results):
                    for result in results:
                        distance = (
                            result.distances[0]
                            if result is not None
                            else np.nan
                        )

                        destination.write(f'{frame}{DELIMITER}{distance}\n')
                        frame += 1

                case _:
                    ...
