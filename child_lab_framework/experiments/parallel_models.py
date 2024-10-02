from collections.abc import AsyncGenerator
from ultralytics import YOLO as Yolo
from ultralytics.engine import results as yolo
import time
from concurrent.futures import ThreadPoolExecutor
import asyncio

from ..util import MODELS_DIR, DEV_DIR
from ..core.video import Reader, Frame, Perspective

# Fakty:
# 1. YOLO zwalnia GIL przy kluczowych obliczeniach
# 2. By osiągnąć równoległość, wątki muszą mieć oddzielne instancje modeli
# 3. ThreadPoolExecutor dobrze łączy się z asyncio i ma akceptowalny narzut


class Detector:
    executor: ThreadPoolExecutor
    model: Yolo

    def __init__(self, executor: ThreadPoolExecutor) -> None:
        self.executor = executor
        self.model = Yolo(MODELS_DIR / 'yolov8x-pose-p6.pt')

    async def stream(
        self,
    ) -> AsyncGenerator[list[yolo.Results] | None, list[Frame] | None]:
        executor = self.executor
        model = self.model
        loop = asyncio.get_running_loop()

        results: list[yolo.Results] | None = None

        while True:
            match (yield results):
                case list(frames):
                    results = await loop.run_in_executor(
                        executor, lambda: model.predict(frames)
                    )

                case _:
                    results = None


async def test() -> None:
    executor = ThreadPoolExecutor(max_workers=2)

    ceiling_reader = Reader(
        str(DEV_DIR / 'data/short/ceiling.mp4'),
        perspective=Perspective.CEILING,
        batch_size=5,
    )
    ceiling = ceiling_reader.stream()

    window_left_reader = Reader(
        str(DEV_DIR / 'data/short/window_left.mp4'),
        perspective=Perspective.CEILING,
        batch_size=5,
    )
    window_left = window_left_reader.stream()

    ceiling_detector = Detector(executor)
    window_detector = Detector(executor)

    ceiling_detector_thread = ceiling_detector.stream()
    await ceiling_detector_thread.asend(None)

    window_detector_thread = window_detector.stream()
    await window_detector_thread.asend(None)

    batch1 = await ceiling.asend(None)
    batch2 = await window_left.asend(None)

    assert batch1 is not None
    assert batch2 is not None

    start = time.time()

    await asyncio.gather(
        ceiling_detector_thread.asend(batch1),
        window_detector_thread.asend(batch2),
    )

    end = time.time()
    elapsed = end - start

    print(f'Done! Elapsed time: {elapsed:.2f} s')


def main() -> None:
    # options = onnx.SessionOptions()
    # options.execution_mode = onnx.ExecutionMode.ORT_PARALLEL
    # options.graph_optimization_level = onnx.GraphOptimizationLevel.ORT_ENABLE_ALL

    # session = onnx.InferenceSession(
    #     str(MODELS_DIR / 'yolov8x-pose-p6.onnx'),
    #     sess_options=options,
    #     providers=['CPUExecutionProvider']
    # )
    #

    asyncio.run(test())
