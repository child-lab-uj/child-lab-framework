import numpy
import viser
from cv2 import aruco as opencv_aruco
from jaxtyping import Float
from marker_detection.aruco import (
    Detector,
    Dictionary,
    MarkerRigidModel,
    VisualizationContext,
)
from video_io.frame import ArrayRgbFrame
from video_io.visualizer import Visualizer

from child_lab_visualization.schema import Frame

__all__ = ['show_image_with_calibration_controls']


def show_image_with_calibration_controls(
    server: viser.ViserServer,
    image: ArrayRgbFrame,
    marker_rigid_model: MarkerRigidModel,
    marker_dictionary: Dictionary,
    marker_detector_parameters: opencv_aruco.DetectorParameters,
) -> None:
    height, width, _ = image.shape
    cx = width / 2.0
    cy = height / 2.0

    visualization_context: VisualizationContext = {
        'intrinsics': numpy.array(()),
        'marker_axis_length': 0.1,
        'marker_axis_thickness': 2,
        'marker_draw_3d_dice_models': True,
        'marker_draw_angles': False,
        'marker_draw_axes': True,
        'marker_draw_ids': False,
        'marker_draw_masks': False,
        'marker_mask_color': (0.0, 255.0, 0.0, 1.0),
        'marker_rigid_model': marker_rigid_model,
    }

    uri = Frame(0)

    height, width, _ = image.shape
    image_handle = server.scene.add_image(
        str(uri.root()),
        image,
        render_width=width,
        render_height=height,
    )

    def replace_displayed_image_with(new_image: ArrayRgbFrame) -> None:
        image_handle.image = new_image

    with server.gui.add_folder('Calibration'):
        focal_length_x_slider = server.gui.add_slider(
            'Focal length (x)',
            2000.0,
            5000.0,
            step=10.0,
            initial_value=3000.0,
        )
        focal_length_y_slider = server.gui.add_slider(
            'Focal length (y)',
            2000.0,
            5000.0,
            step=10.0,
            initial_value=3000.0,
        )

        def intrinsics() -> Float[numpy.ndarray, '3 3']:
            return numpy.array(
                (
                    (focal_length_x_slider.value, 0.0, cx),
                    (0.0, focal_length_y_slider.value, cy),
                    (0.0, 0.0, 1.0),
                )
            )

        def detector() -> Detector:
            return Detector(
                marker_rigid_model,
                marker_dictionary,
                marker_detector_parameters,
                intrinsics(),
                numpy.array((0.0, 0.0, 0.0, 0.0, 0.0)),
            )

        def visualizer() -> Visualizer[VisualizationContext]:
            visualization_context['intrinsics'] = intrinsics()
            return Visualizer(visualization_context)

        focal_length_x_slider.on_update(
            lambda _event: replace_displayed_image_with(
                detect_marker_and_draw_cube(
                    image.copy(),
                    detector(),
                    visualizer(),
                )
            )
        )
        focal_length_y_slider.on_update(
            lambda _event: replace_displayed_image_with(
                detect_marker_and_draw_cube(
                    image.copy(),
                    detector(),
                    visualizer(),
                )
            )
        )


def detect_marker_and_draw_cube(
    frame: ArrayRgbFrame,
    detector: Detector,
    visualizer: Visualizer[VisualizationContext],
) -> ArrayRgbFrame:
    markers = detector.predict(frame)
    if markers is None:
        print('!')
        return frame

    annotated_frame = visualizer.annotate(frame, [markers])
    return annotated_frame
