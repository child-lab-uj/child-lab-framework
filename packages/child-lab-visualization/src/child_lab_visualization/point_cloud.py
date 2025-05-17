from child_lab_data.io.point_cloud import Reader as PointCloudReader
from transformation_buffer.transformation import Transformation
from video_io.reader import Reader as VideoReader
from viser import ViserServer

from child_lab_visualization.schema import Frame


def show_point_cloud(
    server: ViserServer,
    origin_name: str,
    video_reader: VideoReader,
    point_cloud_reader: PointCloudReader,
    transformation: Transformation = Transformation.identity(),
) -> None:
    uri = Frame(0)

    point_cloud = point_cloud_reader.read()
    assert point_cloud is not None

    frame = video_reader.read()
    assert frame is not None

    points = point_cloud.permute((1, 2, 0)).flatten(0, -2).numpy()
    colors = frame.permute((1, 2, 0)).flatten(0, -2).numpy()

    server.scene.add_point_cloud(
        str(uri.point_cloud(origin_name)),
        points=points,
        colors=colors,
        point_size=0.001,
        point_shape='circle',
        wxyz=transformation.rotation_quaternion().numpy(),
        position=transformation.translation().numpy(),
    )
