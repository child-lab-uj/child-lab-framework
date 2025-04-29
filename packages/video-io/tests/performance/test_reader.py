from pathlib import Path
from typing import Literal, Protocol

import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture
from video_io.reader import Reader

from ..helpers import TEST_DATA_DIRECTORY
from .opencv_baseline import PoorMansReader


@pytest.fixture(scope='module')
def calibration_video() -> Path:
    return TEST_DATA_DIRECTORY / 'calibration' / 'lab_ceiling.avi'


@pytest.mark.benchmark(
    group='reader-benchmark',
    disable_gc=True,
    min_rounds=10,
)
@pytest.mark.skip(reason='Performance test is not a part of the standard suite.')
@pytest.mark.usefixtures('calibration_video')
@pytest.mark.parametrize('batch_size', [10, 30, 50])
@pytest.mark.parametrize('reader_type', ['torch', 'opencv'])
def test_reader(
    benchmark: BenchmarkFixture,
    calibration_video: Path,
    batch_size: int,
    reader_type: Literal['torch', 'opencv'],
) -> None:
    reader = (
        Reader(calibration_video, torch.device('cpu'))
        if reader_type == 'torch'
        else PoorMansReader(calibration_video)
    )
    benchmark(lambda: read_whole_video(reader, batch_size))


class AnyReader(Protocol):
    def read_batch(self, size: int) -> object | None: ...


def read_whole_video(reader: AnyReader, batch_size: int) -> None:
    while True:
        batch = reader.read_batch(batch_size)

        if batch is None:
            break
