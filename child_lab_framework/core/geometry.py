import torch
import typing
from collections.abc import Callable


def area(rect: torch.Tensor) -> torch.Tensor:
    width = rect[3] - rect[1]
    height = rect[2] - rect[0]
    return width * height

area_broadcast = typing.cast(
    Callable[[torch.Tensor], torch.Tensor],
    torch.vmap(area, 0)
)
