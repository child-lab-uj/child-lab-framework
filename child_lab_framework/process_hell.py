import sys
from .core.video import Reader, Writer, Perspective, Properties, Format
from .task import pose, face, gaze
from .task.visualization import Visualizer
import os
import numpy as np
import asyncio
from functools import wraps
import time
from typing import Callable, Any
from collections.abc import Iterable, Mapping, AsyncGenerator
from multiprocessing import Process, Manager
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager, ValueProxy
from concurrent.futures import ProcessPoolExecutor
from ultralytics import YOLO
from .util import MODELS_DIR
from .core.video import Frame
from threading import Semaphore
import signal
import psutil


class YoloServer(Process):
    name: str
    model: YOLO
    input: SharedMemory
    output: SharedMemory
    status: SharedMemory

    def __init__(
        self,
        input: SharedMemory,
        output: SharedMemory,
        status: SharedMemory,
        name: str | None = None,
    ) -> None:
        super().__init__(None, None, name, (), {}, daemon=None)

        self.input = input
        self.output = output
        self.status = status

        self.name = name or 'dupa'

        self.model = YOLO(str(MODELS_DIR / 'yolov8x-pose-p6.pt'))

    def run(self) -> None:
        super().run()

        input = np.ndarray((1080, 1920, 3), dtype=np.uint8, buffer=self.input.buf)
        output = np.ndarray((2, 17, 6), dtype=np.float32, buffer=self.output.buf)

        me = psutil.Process(self.pid)

        print(f'[Worker] Started')

        while True:
            me.suspend()
            print(f'[Worker] Unpaused')

            result = self.model.predict(input, verbose=False)

            if len(result) == 0:
                continue

            print(f'[Worker] {result[0].keypoints = }')

            first_keypoints = result[0].keypoints.data.numpy()
            n_people, n_keypoints, n_coordinates = first_keypoints.shape

            self.status.buf[0] = 1
            self.status.buf[1] = n_people
            self.status.buf[2] = n_keypoints
            self.status.buf[3] = n_coordinates

            output[:n_people, :n_keypoints, :n_coordinates] = first_keypoints

            print(f'[Worker] Processed frame')


async def execute(
    input: SharedMemory,
    output: SharedMemory,
    status: SharedMemory,
    name: str | None = None,
    wait_interval: float = 0.01
) -> AsyncGenerator[Frame | None, np.ndarray | None]:
    results: np.ndarray | None = None

    input_array = np.ndarray((1080, 1920, 3), dtype=np.uint8, buffer=input.buf)
    output_array = np.ndarray((2, 17, 3), dtype=np.float32, buffer=output.buf)
    output_array.flags.writeable = False

    status.buf[0] = 0
    status.buf[1] = 0
    status.buf[2] = 0
    status.buf[3] = 0

    worker = YoloServer(input, output, status, name)
    worker.start()

    worker_handler = psutil.Process(worker.pid)

    await asyncio.sleep(1.0)

    while True:
        match (yield results):
            case None:
                results = None

            case frame:
                np.copyto(input_array, frame)
                worker_handler.resume()
                print(f'Sent SIGCONT')

                while status.buf[0] == 0:
                    print('Waiting...')
                    await asyncio.sleep(wait_interval)

                status.buf[0] = 0
                n_people, n_keypoints, n_coordinates = status.buf[1:4]

                results = output_array[:n_people, :n_keypoints, :n_coordinates]


async def _main() -> None:
    smm = SharedMemoryManager()
    smm.start()

    input = smm.SharedMemory(1920 * 1080 * 3)
    output = smm.SharedMemory(2 * 17 * 6 * 32)
    status = smm.SharedMemory(4 * 32)

    gen = execute(input, output, status, 'dupa', 0.5)
    await gen.asend(None)

    ceiling_reader = Reader(
        'dev/data/ultra_short/ceiling.mp4',
        perspective=Perspective.CEILING,
        batch_size=1
    )

    ceiling_reader_thread = ceiling_reader.stream()

    res = await gen.asend(ceiling_reader_thread.send(None)[0])

    print(f'Received:\n{res}')

    smm.shutdown()


def main() -> None:
    asyncio.run(_main())
