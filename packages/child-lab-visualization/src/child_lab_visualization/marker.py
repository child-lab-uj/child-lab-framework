from transformation_buffer.transformation import Transformation
from viser import ViserServer

from child_lab_visualization.schema import Frame


def show_marker(
    server: ViserServer,
    name: str,
    transformation: Transformation,
) -> None:
    uri = Frame(0)

    server.scene.add_frame(
        str(uri.frame_of_reference(name)),
        axes_length=0.1,
        axes_radius=0.0025,
        wxyz=transformation.rotation_quaternion().numpy(),
        position=transformation.translation().numpy(),
        visible=False,
    )
