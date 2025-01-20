import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy
import torch
from depth_pro import Config, DepthPro
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, Resize

from ...core.video import Properties
from ...logging import Logger
from ...postprocessing.imputation import imputed_with_closest_known_reference
from ...typing.array import FloatArray2
from ...typing.stream import Fiber
from ...typing.video import Frame
from ...util import MODELS_DIR


def to_frame(depth_map: FloatArray2) -> Frame:
    normalized = (depth_map / depth_map.max() * 255).astype(numpy.uint8)
    return numpy.transpose(numpy.stack((normalized, normalized, normalized)), (1, 2, 0))


type Input = tuple[list[Frame] | None, list[Properties] | None]


class Estimator:
    MODEL_PATH = MODELS_DIR / 'depth_pro.pt'

    executor: ThreadPoolExecutor | None
    device: torch.device

    model: DepthPro
    model_config: Config

    input: Properties

    to_model: Compose

    def __init__(
        self,
        device: torch.device,
        *,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self.executor = executor
        self.device = device

        config = Config(checkpoint=self.MODEL_PATH)
        self.model_config = config
        self.model = DepthPro(config, device, torch.half)

        Logger.info('Depth model created')

        self.to_model = Compose(  # type: ignore[no-untyped-call]
            [
                ConvertImageDtype(torch.half),
                Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # type: ignore[no-untyped-call]
            ]
        )

    def predict(self, frame: Frame, properties: Properties) -> FloatArray2:
        input = torch.from_numpy(frame.copy()).to(self.device)
        input = torch.permute(input, (2, 0, 1))
        input = torch.unsqueeze(input, 0)
        input = self.to_model(input)

        # shape of the input after transposition: 1 x n_channels x height x width

        focal_length = (
            properties.calibration.focal_length[0]
            * self.model.input_image_size
            / properties.width
        )

        # TODO: return the tensor itself without transferring (Issue #6)
        result = self.model.predict(input, focal_length)
        depth: FloatArray2 = (
            Resize((properties.height, properties.width))  # type: ignore[no-untyped-call]
            .forward(result.depth)
            .to(torch.float32)
            .mul(1000.0)
            .cpu()
            .numpy()
        )
        del result

        return depth

    async def stream(self) -> Fiber[Input | None, list[FloatArray2] | None]:
        executor = self.executor
        if executor is None:
            raise RuntimeError(
                'Processing in the stream mode requires the Estimator to have an executor. Please pass an "executor" argument to the estimator constructor'
            )

        loop = asyncio.get_running_loop()

        results: list[FloatArray2] | None = None

        while True:
            match (yield results):
                case list(frames), list(properties):
                    n_frames = len(frames)

                    results = await loop.run_in_executor(
                        executor,
                        lambda: imputed_with_closest_known_reference(
                            [
                                self.predict(
                                    frames[n_frames // 2], properties[n_frames // 2]
                                )
                            ]
                            + [None for _ in range(n_frames - 1)]
                        ),
                    )

                    Logger.info('Depth estimated')

                case _:
                    results = None
