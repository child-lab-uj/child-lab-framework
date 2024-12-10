from collections.abc import Sequence
from enum import Enum
from pathlib import Path

import polars
from more_itertools import unique

from ...core.dumping import Dumpable


class Format(Enum):
    CSV = 'csv'
    PARQUET = 'parquet'


class Dumper:
    destination: Path
    format: Format

    __data_frame: polars.DataFrame
    __current_video_frame: int

    def __init__(self, destination: Path, format: Format) -> None:
        self.destination = destination
        self.format = format
        self.__data_frame = polars.DataFrame(schema={'frame': polars.Int64})
        self.__current_video_frame = 0

    def __del__(self) -> None:
        self.flush()

    def flush(self) -> None:
        match self.format:
            case Format.CSV:
                self.__data_frame.write_csv(self.destination)

            case Format.PARQUET:
                self.__data_frame.write_parquet(self.destination)

    def dump(self, result: Dumpable) -> None:
        current_frame = self.__current_video_frame

        dumped = result.data_frame
        dumped.insert_column(0, polars.Series('frame', [current_frame]))

        self.__data_frame.extend(dumped)
        self.__current_video_frame += 1

    def dump_batch[T: Dumpable](self, results: Sequence[T | None]) -> None:
        batch_length = len(results)
        if batch_length == 0:
            return

        if all(result is None for result in results):
            return

        current_frame = self.__current_video_frame

        dumped = polars.concat(
            [
                self.__dump_result_with_frame_number(result, current_frame)
                for result in results
                if result is not None
            ],
            how='vertical',
        )

        self.__data_frame.extend(dumped)
        self.__current_video_frame += batch_length

    def dump_batches(self, *result_batches: Sequence[Dumpable | None]) -> None:
        batch_lengths = list(unique(map(len, result_batches)))

        match len(batch_lengths):
            case 0:
                return

            case 1:
                ...

            case _:
                raise ValueError(
                    f'Expected all batches to have identical size, got {batch_lengths}'
                )

        # no way to enforce the type homogeneity of the batches on the type system level - need runtime check :c
        batch_types = [
            [type(element) for element in batch if element is not None]
            for batch in result_batches
        ]

        non_homogeneous_batch_types = [
            (i, types)
            for (i, types) in enumerate(batch_types)
            if len(list(unique(types))) != 1
        ]

        if len(non_homogeneous_batch_types) > 0:
            punctuated = '\n'.join(
                f'{i} with types {types},' for i, types in non_homogeneous_batch_types
            )

            raise ValueError(
                f'Expected every batch to be type-homogeneous. Got:\n{punctuated}'
            )

        batch_length = batch_lengths[0]
        current_frame = self.__current_video_frame

        dumped_batches = polars.concat(
            [
                polars.concat(
                    [
                        self.__dump_result_with_frame_number(result, frame)
                        for result in frame_results
                        if result is not None
                    ],
                    how='horizontal',
                )
                for (frame, *frame_results) in zip(
                    range(current_frame, current_frame + batch_length),
                    *result_batches,
                )
            ],
            how='vertical_relaxed',  # allow nulls
        )

        self.__data_frame.extend(dumped_batches)
        self.__current_video_frame += batch_length

    @staticmethod
    def __dump_result_with_frame_number(result: Dumpable, frame: int) -> polars.DataFrame:
        dumped_result = result.data_frame.clone()
        dumped_result.insert_column(
            0,
            polars.Series('frame', [frame for _ in range(dumped_result.shape[0])]),
        )

        return dumped_result
