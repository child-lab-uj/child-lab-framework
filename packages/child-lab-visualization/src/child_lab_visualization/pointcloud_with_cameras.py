import numpy
from child_lab_data.io.point_cloud import Reader as PointCloudReader
from transformation_buffer.buffer import Buffer
from video_io.calibration import Calibration
from video_io.reader import Reader
from viser import ViserServer

from child_lab_visualization.schema import Frame


def show_pointcloud_and_camera_poses(
    server: ViserServer,
    origin_name: str,
    reader: Reader,
    point_cloud_reader: PointCloudReader,
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
                position=transformation.translation().numpy(),
            )

        server.scene.add_frame(
            str(uri.frame_of_reference(frame)),
            axes_length=0.1,
            axes_radius=0.0025,
            wxyz=transformation.rotation_quaternion().numpy(),
            position=transformation.translation().numpy(),
            visible=False,
        )

    point_cloud = point_cloud_reader.read()
    assert point_cloud is not None

    frame = reader.read()
    assert frame is not None

    points = point_cloud.permute((1, 2, 0)).flatten(0, -2).numpy()
    colors = frame.permute((1, 2, 0)).flatten(0, -2).numpy()

    server.scene.add_point_cloud(
        str(uri.point_cloud(origin_name)),
        points=points,
        colors=colors,
        point_size=0.001,
        point_shape='circle',
        position=(0.0, 0.0, -2.0),
    )
