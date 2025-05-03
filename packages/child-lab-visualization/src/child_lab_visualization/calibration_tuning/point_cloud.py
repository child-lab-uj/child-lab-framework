import cv2 as opencv
import numpy
import torch
import viser
from depth_estimation.depth_pro import DepthPro
from marker_detection.aruco import Detector, Dictionary, MarkerRigidModel
from video_io.calibration import Calibration
from video_io.frame import ArrayRgbFrame, TensorRgbFrame

DEFAULT_FOCAL_LENGTH = 3000.0


def show_point_cloud_with_calibration_controls(
    server: viser.ViserServer,
    image: TensorRgbFrame,
    depth_estimator: DepthPro,
    marker_rigid_model: MarkerRigidModel,
    marker_dictionary: Dictionary,
    marker_detector_parameters: opencv.aruco.DetectorParameters,
) -> None:
    with server.gui.add_folder('Calibration'):
        depth_pro_fx_checkbox = server.gui.add_checkbox(
            'Use DepthPro FOV estimate',
            False,
        )
        fx_slider = server.gui.add_slider(
            'Focal length (x)',
            2000.0,
            5000.0,
            step=10.0,
            initial_value=DEFAULT_FOCAL_LENGTH,
        )
        fy_slider = server.gui.add_slider(
            'Focal length (y)',
            2000.0,
            5000.0,
            step=10.0,
            initial_value=DEFAULT_FOCAL_LENGTH,
        )

        recompute_button = server.gui.add_button('Recompute')

    _, height, width = image.shape
    cx = width / 2.0
    cy = height / 2.0
    fov = 2 * numpy.arctan2(height / 2, DEFAULT_FOCAL_LENGTH)
    aspect = width / height

    array_image: ArrayRgbFrame = image.permute((1, 2, 0)).cpu().numpy()

    camera_frustum = server.scene.add_camera_frustum('/views/origin/camera', fov, aspect)

    starting_points = (
        Calibration(
            (DEFAULT_FOCAL_LENGTH, DEFAULT_FOCAL_LENGTH),
            (cx, cy),
            (0.0, 0.0, 0.0, 0.0, 0.0),
        )
        .unproject_depth(torch.ones(height, width))
        .permute((1, 2, 0))
        .flatten(0, -2)
        .numpy()
    )
    colors = image.permute((1, 2, 0)).cpu().flatten(0, -2).numpy()

    point_cloud = server.scene.add_point_cloud(
        '/views/origin/point_cloud',
        points=starting_points,
        colors=colors,
        point_size=0.001,
        point_shape='circle',
        position=(0.0, 0.0, 0.0),
    )

    def update() -> None:
        if depth_pro_fx_checkbox.value:
            estimate = depth_estimator.predict(image)
            fx = fy = estimate.focal_length_px
            new_depth = estimate.depth.cpu()
        else:
            fx = fx_slider.value
            fy = fy_slider.value
            new_depth = depth_estimator.predict(image, fx).depth.cpu()

        new_points = (
            Calibration(
                (fx, fy),
                (cx, cy),
                (0.0, 0.0, 0.0, 0.0, 0.0),
            )
            .unproject_depth(new_depth)
            .permute((1, 2, 0))
            .flatten(0, -2)
            .numpy()
        )

        point_cloud.points = new_points

        new_fov = 2 * numpy.arctan2(height / 2, fx)
        camera_frustum.fov = new_fov

        intrinsics_matrix = numpy.array(
            (
                (fx, 0.0, cx),
                (0.0, fy, cy),
                (0.0, 0.0, 1.0),
            )
        )

        markers = Detector(
            marker_rigid_model,
            marker_dictionary,
            marker_detector_parameters,
            intrinsics_matrix,
            numpy.array((0.0, 0.0, 0.0, 0.0, 0.0)),
        ).predict(array_image)

        if markers is not None:
            for id, transformation in zip(markers.ids, markers.transformations):
                server.scene.add_frame(
                    f'/views/origin/markers/{id}',
                    axes_length=0.1,
                    axes_radius=0.0025,
                    wxyz=transformation.rotation_quaternion(),
                    position=transformation.translation,
                )

    recompute_button.on_click(lambda _event: update())
