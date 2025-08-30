from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import kornia
import torch
from child_lab_data.io.point_cloud import Writer as PointCloudWriter
from jaxtyping import UInt8, UInt16
from video_io.calibration import Calibration
from video_io.writer import Writer as VideoWriter
from web_camera.oak_d import Reader

__all__ = ['IoContext', 'Procedure']


@dataclass
class IoContext:
    calibration: Calibration
    camera_reader: Reader

    video_writer: VideoWriter
    depth_map_writer: VideoWriter
    point_cloud_writer: PointCloudWriter


class Procedure:
    context: IoContext

    def __init__(self, context: IoContext) -> None:
        self.context = context

    def run(self, on_step: Callable[[], Any] | None = None) -> None:
        camera_reader = self.context.camera_reader
        video_writer = self.context.video_writer
        depth_map_writer = self.context.depth_map_writer
        point_cloud_writer = self.context.point_cloud_writer
        calibration = camera_reader.calibration()

        while True:
            if on_step is not None:
                on_step()

            frame, depth = camera_reader.read()
            point_cloud = calibration.unproject_depth(depth)
            depth_image = colorize_depth(depth)

            video_writer.write(frame)
            depth_map_writer.write(depth_image)
            point_cloud_writer.write(point_cloud)


def colorize_depth(
    depth: UInt16[torch.Tensor, 'height width'],
) -> UInt8[torch.Tensor, '3 height width']:
    height, width = depth.shape

    depth = depth.to(torch.float32).reshape((1, 1, height, width))
    depth = kornia.enhance.normalize_min_max(depth, min_val=0.0, max_val=1.0)
    depth = kornia.enhance.equalize(depth)
    depth = kornia.filters.gaussian_blur2d(depth, (5, 5), (1.0, 1.0))
    depth = kornia.color.apply_colormap(depth, kornia.color.ColorMap('viridis'))
    depth *= 255.0
    depth = depth.squeeze().to(torch.uint8)

    return depth
