from dataclasses import dataclass

import serde
import torch
from jaxtyping import Float, Float64


@serde.serde
@dataclass(slots=True)
class Calibration:
    focal_length: tuple[float, float]
    optical_center: tuple[float, float]
    distortion: tuple[float, float, float, float, float]

    def intrinsics_matrix(self) -> Float64[torch.Tensor, '3 3']:
        output = torch.zeros((3, 3), dtype=torch.float64)

        fx, fy = self.focal_length
        cx, cy = self.optical_center

        output[0, 0] = fx
        output[1, 1] = fy
        output[0, 2] = cx
        output[1, 2] = cy
        output[2, 2] = 1.0

        return output

    def distortion_vector(self) -> Float64[torch.Tensor, '5']:
        return torch.tensor(self.distortion, dtype=torch.float64)

    def unproject_depth(
        self,
        depth: Float[torch.Tensor, 'height width'],
    ) -> Float[torch.Tensor, '3 height width']:
        *_, height, width = depth.shape

        u = torch.arange(width)
        v = torch.arange(height)
        u, v = torch.meshgrid(u, v, indexing='xy')

        fx, fy = self.focal_length
        cx, cy = self.optical_center

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return torch.stack((x, y, z))
