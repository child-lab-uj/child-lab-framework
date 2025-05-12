import torch
from jaxtyping import Float, Int


def average_depth_at(
    depth: Float[torch.Tensor, 'height width'],
    coordinates: Int[torch.Tensor, 'n_points 2'],
    radius: int,
) -> Float[torch.Tensor, ' n_points']:
    assert depth.device == coordinates.device

    neighborhood = torch.arange(-radius, radius).reshape(1, 1, -1).to(depth.device)

    height, width = depth.shape
    extended_coordinates = coordinates.unsqueeze(-1) + neighborhood

    depth_values = depth[
        extended_coordinates[:, 1, :].clamp(0, height - 1),
        extended_coordinates[:, 0, :].clamp(0, width - 1),
    ]

    average_depth = depth_values.mean(dim=1)

    return average_depth
