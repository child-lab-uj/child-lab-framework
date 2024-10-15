import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
import torch
from depth_pro import depth_pro as dp
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, Resize

from ...core.video import Frame, Properties
from ...logging import Logger
from ...typing.array import FloatArray2
from ...typing.stream import Fiber
from ...util import MODELS_DIR

type DepthProResult = dict[Literal['depth', 'focallength_px'], torch.Tensor]


def to_frame(depth_map: FloatArray2) -> Frame:
    green = (depth_map / depth_map.max() * 255).astype(np.uint8)
    other = np.zeros_like(green)
    frame = np.transpose(np.stack((other, green, other)), (1, 2, 0))
    return frame


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

        self.model, _ = dp.create_model_and_transforms(config, device)

        self.to_model = Compose(
            [
                ConvertImageDtype(torch.float32),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        config = dp.DEFAULT_MONODEPTH_CONFIG_DICT
        config.checkpoint_uri = self.MODEL_PATH

        self.model, _ = dp.create_model_and_transforms(config, device)

        self.to_model = Compose(
            [
                ConvertImageDtype(torch.float32),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.from_model = Compose([Resize((input.height, input.width))])

    def predict(self, frame: Frame) -> FloatArray2:
        # Frame's immutability causes runtime warnings (Probably to fix in the issue #6)
        input = torch.from_numpy(frame).to(self.device)
        input = torch.permute(input, (2, 0, 1))
        input = torch.unsqueeze(input, 0)
        input = self.to_model(input)

        # shape of the input after transposition: 1 x n_channels x height x width
        result: DepthProResult = self.model.infer(input)  # type: ignore  # That's why returning a `Mapping` isn't a good idea :v

        depth = self.from_model(result['depth'])

        Logger.info('Depth estimated')

        # TODO: return the tensor itself without transferring (Issue #6)
        return depth.cpu().numpy()

    async def stream(self) -> Fiber[list[Frame] | None, list[FloatArray2] | None]:
        loop = asyncio.get_running_loop()
        executor = self.executor

        results: list[FloatArray2] | None = None

        while True:
            match (yield results):
                case list(frames):
                    results = await loop.run_in_executor(
                        executor, lambda: [self.predict(frame) for frame in frames]
                    )

                case _:
                    results = None
