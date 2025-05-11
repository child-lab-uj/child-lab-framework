from pathlib import Path

import numpy
import torch
from transformation_buffer.buffer import Buffer
from video_io.calibration import Calibration
from video_io.reader import Reader
from viser import ViserServer

from child_lab_visualization.schema import Frame


def show_pointcloud_and_camera_poses(
    server: ViserServer,
    origin_name: str,
    reader: Reader,
    points_files: list[Path],
    calibration: Calibration,
    buffer: Buffer,
) -> None:
    height = reader.metadata.height
    width = reader.metadata.width
    fov = 2 * numpy.arctan2(height / 2, calibration.focal_length[0])
    aspect = width / height

    uri = Frame(0)

    server.scene.add_camera_frustum(str(uri.camera(origin_name)), fov, aspect)

    for frame in buffer.frames_visible_from(origin_name):
        transformation = buffer[(frame, origin_name)]
        assert transformation is not None

        if 'marker' not in frame:  # => it is a camera
            server.scene.add_camera_frustum(
                str(uri.camera(frame)),
                fov,
                aspect,
                wxyz=transformation.rotation_quaternion().numpy(),
                position=transformation.translation().div(1000.0).numpy(),
            )

        server.scene.add_frame(
            str(uri.frame_of_reference(frame)),
            axes_length=0.1,
            axes_radius=0.0025,
            wxyz=transformation.rotation_quaternion().numpy(),
            # Dividing by 1000 is a workaround for working with old intrinsic parameters.
            position=transformation.translation().div(1000.0).numpy(),
            visible=False,
        )

    points_first_batch: torch.Tensor = torch.load(points_files[0])
    frames_first_batch = reader.read_batch(points_first_batch.shape[0])
    assert frames_first_batch is not None

    server.scene.add_point_cloud(
        str(uri.point_cloud(origin_name)),
        points=points_first_batch[0].permute((1, 2, 0)).flatten(0, -2).numpy(),
        colors=frames_first_batch[0].permute((1, 2, 0)).flatten(0, -2).numpy(),
        point_size=0.001,
        point_shape='circle',
        position=(0.0, 0.0, -2.0),
    )
