import numpy
from transformation_buffer.transformation import Transformation
from video_io.calibration import Calibration
from video_io.metadata import Metadata
from viser import ViserServer

from child_lab_visualization.schema import Frame


def show_camera(
    server: ViserServer,
    name: str,
    video_properties: Metadata,
    calibration: Calibration,
    transformation: Transformation,
) -> None:
    height = video_properties.height
    width = video_properties.width
    fov = 2 * numpy.arctan2(height / 2, calibration.focal_length[0])
    aspect = width / height

    uri = Frame(0)

    server.scene.add_camera_frustum(
        str(uri.camera(name)),
        fov,
        aspect,
        wxyz=transformation.rotation_quaternion().numpy(),
        position=transformation.translation().numpy(),
    )
