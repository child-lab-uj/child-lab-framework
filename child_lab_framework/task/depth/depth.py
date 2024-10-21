import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy
import torch
from depth_pro import depth_pro as dp
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, Resize

from ...core.sequence import imputed_with_reference_inplace
from ...core.video import Frame, Properties
from ...logging import Logger
from ...typing.array import FloatArray2
from ...typing.stream import Fiber
from ...util import MODELS_DIR

type DepthProResult = dict[Literal['depth', 'focallength_px'], torch.Tensor]


def to_frame(depth_map: FloatArray2) -> Frame:
    green = (depth_map / depth_map.max() * 255).astype(numpy.uint8)
    other = numpy.zeros_like(green)
    frame = numpy.transpose(numpy.stack((other, green, other)), (1, 2, 0))
    return frame


# DEVICE_LOCK: threading.Lock = threading.Lock()


class Estimator:
    MODEL_PATH = str(MODELS_DIR / 'depth_pro.pt')
    MODEL_INPUT_SIZE = (616, 1064)
    PADDING_BORDER_COLOR = (123.675, 116.28, 103.53)

    executor: ThreadPoolExecutor
    device: torch.device
    model: dp.DepthPro

    input: Properties

    to_model: Compose
    from_model: Compose

    def __init__(
        self, executor: ThreadPoolExecutor, device: torch.device, *, input: Properties
    ) -> None:
        self.executor = executor
        self.device = device
        self.input = input

        config = dp.DEFAULT_MONODEPTH_CONFIG_DICT
        config.checkpoint_uri = self.MODEL_PATH

        memory_usage = float(torch.mps.current_allocated_memory()) / 1024.0 / 1024.0

        self.model, _ = dp.create_model_and_transforms(
            config, device, precision=torch.half
        )

        memory_usage_after_creation = (
            float(torch.mps.current_allocated_memory()) / 1024.0 / 1024.0
        )

        Logger.info(
            f'Depth model created; memory usage: before: {memory_usage:.2f} MiB, after: {memory_usage_after_creation:.2f} MiB'
        )

        self.to_model = Compose(
            [
                ConvertImageDtype(torch.half),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.from_model = Compose(
            [
                ConvertImageDtype(torch.float32),
                Resize((input.height, input.width)),
            ]
        )

    def predict(self, frame: Frame) -> FloatArray2:
        # frame = opencv.cvtColor(frame, opencv.COLOR_BGR2RGB)  # type: ignore

        input = torch.from_numpy(frame.copy()).to(self.device)
        input = torch.permute(input, (2, 0, 1))
        input = torch.unsqueeze(input, 0)
        input = self.to_model(input)

        # shape of the input after transposition: 1 x n_channels x height x width

        # TODO: return the tensor itself without transferring (Issue #6)
        result: DepthProResult = self.model.infer(input)  # type: ignore  # That's why returning a `Mapping` isn't a good idea :v
        depth = self.from_model(result['depth']).cpu().numpy()
        del result

        return depth

    async def stream(self) -> Fiber[list[Frame] | None, list[FloatArray2] | None]:
        loop = asyncio.get_running_loop()
        executor = self.executor

        results: list[FloatArray2] | None = None

        while True:
            match (yield results):
                case list(frames):
                    n_frames = len(frames)

                    results = await loop.run_in_executor(
                        executor,
                        lambda: imputed_with_reference_inplace(
                            [self.predict(frames[n_frames // 2])]
                            + [None for _ in range(n_frames - 1)]
                        ),
                    )

                    Logger.info('Depth estimated')

                case _:
                    results = None
