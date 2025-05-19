import logging
from dataclasses import dataclass
from pathlib import Path

import numpy
import torch
import viser
from child_lab_procedures.estimate_transformations import (
    Configuration,
    Procedure,
    VideoIoContext,
)
from depth_estimation.depth_pro import DepthPro
from marker_detection.aruco import Dictionary, MarkerRigidModel
from transformation_buffer.buffer import Buffer
from transformation_buffer.rigid_model import Cube
from transformation_buffer.transformation import Transformation
from video_io import Reader as VideoReader
from video_io.calibration import Calibration
from video_io.frame import TensorRgbFrame

from child_lab_visualization.schema import Frame

__all__ = ['Input', 'show_point_cloud_with_calibration_controls']

DEFAULT_FOCAL_LENGTH = 3_000.0
FOCAL_LENGTH_SLIDER_MIN = 1_000.0
FOCAL_LENGTH_SLIDER_MAX = 10_000.0


@dataclass(slots=True)
class Input:
    name: str
    location: Path
    first_frame: TensorRgbFrame


def show_point_cloud_with_calibration_controls(
    server: viser.ViserServer,
    inputs: list[Input],
    depth_estimator: DepthPro,
    marker_rigid_model: MarkerRigidModel,
    marker_dictionary: Dictionary,
    arudice: list[Cube[str]],
) -> None:
    perspectives = {
        input.name: add_perspective(server, input.name, input.first_frame)
        for input in inputs
    }

    estimate_transformations_configuration = Configuration(
        marker_rigid_model,
        marker_dictionary,
        arudice=arudice,
    )

    server.gui.add_button('Refresh').on_click(lambda _event: update())

    def estimate_transformations() -> Buffer | None:
        contexts = [
            VideoIoContext(
                input.name,
                perspectives[input.name].update.calibration(),
                VideoReader(input.location),
            )
            for input in inputs
        ]
        procedure = Procedure(estimate_transformations_configuration, contexts)
        return procedure.run()

    def update() -> None:
        for perspective in perspectives.values():
            if perspective.needs_refresh:
                logging.debug(f'Recomputing depth for {perspective.name}')

                if perspective.ui.depth_pro_fx_checkbox.value:
                    estimate = depth_estimator.predict(perspective.image)
                    fx = fy = estimate.focal_length_px
                    new_depth = estimate.depth.cpu()
                else:
                    fx = perspective.ui.fx_slider.value
                    fy = perspective.ui.fy_slider.value
                    new_depth = depth_estimator.predict(perspective.image, fx).depth.cpu()

                perspective.update.depth = new_depth
                perspective.update.fx = fx
                perspective.update.fy = fy

        buffer = estimate_transformations()
        if buffer is None:
            logging.warning('Transformation estimation failed!')
            return None

        origin = list(perspectives.keys())[0]
        origin_perspective = perspectives[origin]

        for name, perspective in perspectives.items():
            transformation = buffer[(name, origin)]
            if transformation is None:
                continue

            perspective.update.transformation = transformation
            perspective.refresh_ui()

        for name in buffer.frames_of_reference:
            if 'marker' not in name:
                continue

            transformation = buffer[(name, origin)]
            if transformation is None:
                continue

            server.scene.add_frame(
                str(origin_perspective.frame_uri.frame_of_reference(name)),
                axes_length=0.1,
                axes_radius=0.0025,
                wxyz=transformation.rotation_quaternion().numpy(),
                position=transformation.translation().numpy(),
            )


@dataclass(slots=True)
class PerspectiveUi:
    depth_pro_fx_checkbox: viser.GuiCheckboxHandle
    fx_slider: viser.GuiSliderHandle
    fy_slider: viser.GuiSliderHandle
    camera_frustum: viser.CameraFrustumHandle
    point_cloud: viser.PointCloudHandle


@dataclass(slots=True)
class PerspectiveUpdate:
    transformation: Transformation
    depth: torch.Tensor
    fx: float
    fy: float
    cx: float
    cy: float

    def calibration(self) -> Calibration:
        return Calibration(
            (self.fx, self.fy),
            (self.cx, self.cy),
            (0.0, 0.0, 0.0, 0.0, 0.0),
        )


@dataclass(slots=True)
class Perspective:
    name: str
    image: TensorRgbFrame
    needs_refresh: bool
    update: PerspectiveUpdate
    ui: PerspectiveUi
    frame_uri: Frame

    def refresh_ui(self) -> None:
        calibration = self.update.calibration()
        point_cloud = calibration.unproject_depth(self.update.depth)
        self.ui.point_cloud.points = (
            point_cloud.permute((1, 2, 0)).flatten(0, -2).cpu().numpy()
        )

        transformation = self.update.transformation
        wxyz = transformation.rotation_quaternion().numpy()
        position = transformation.translation().numpy()

        self.ui.camera_frustum.wxyz = wxyz
        self.ui.camera_frustum.position = position
        self.ui.point_cloud.wxyz = wxyz
        self.ui.point_cloud.position = position

        new_fov = 2 * numpy.arctan2(self.image.shape[1] / 2, self.update.fx)
        self.ui.camera_frustum.fov = new_fov

        self.needs_refresh = False


def add_perspective(
    server: viser.ViserServer,
    name: str,
    first_frame: TensorRgbFrame,
) -> Perspective:
    with server.gui.add_folder(f'{name} calibration'):
        depth_pro_fx_checkbox = server.gui.add_checkbox(
            'Use DepthPro FOV estimate',
            False,
        )
        fx_slider = server.gui.add_slider(
            'Focal length (x)',
            FOCAL_LENGTH_SLIDER_MIN,
            FOCAL_LENGTH_SLIDER_MAX,
            step=10.0,
            initial_value=DEFAULT_FOCAL_LENGTH,
        )
        fy_slider = server.gui.add_slider(
            'Focal length (y)',
            FOCAL_LENGTH_SLIDER_MIN,
            FOCAL_LENGTH_SLIDER_MAX,
            step=10.0,
            initial_value=DEFAULT_FOCAL_LENGTH,
        )

    _, height, width = first_frame.shape
    cx = width / 2.0
    cy = height / 2.0
    fov = 2 * numpy.arctan2(height / 2, DEFAULT_FOCAL_LENGTH)
    aspect = width / height

    uri = Frame(0)

    transformation = Transformation.identity()

    camera_frustum = server.scene.add_camera_frustum(
        str(uri.camera(name)),
        fov,
        aspect,
        wxyz=transformation.rotation_quaternion().numpy(),
        position=transformation.translation().numpy(),
    )

    starting_depth = torch.ones(height, width)
    starting_points = (
        Calibration(
            (DEFAULT_FOCAL_LENGTH, DEFAULT_FOCAL_LENGTH),
            (cx, cy),
            (0.0, 0.0, 0.0, 0.0, 0.0),
        )
        .unproject_depth(starting_depth)
        .permute((1, 2, 0))
        .flatten(0, -2)
        .numpy()
    )
    colors = first_frame.permute((1, 2, 0)).cpu().flatten(0, -2).numpy()

    point_cloud = server.scene.add_point_cloud(
        str(uri.point_cloud(name)),
        points=starting_points,
        colors=colors,
        point_size=0.001,
        point_shape='circle',
        wxyz=transformation.rotation_quaternion().numpy(),
        position=transformation.translation().numpy(),
    )

    perspective = Perspective(
        name,
        first_frame,
        needs_refresh=True,
        frame_uri=uri,
        update=PerspectiveUpdate(
            transformation=Transformation.identity(),
            depth=starting_depth,
            fx=DEFAULT_FOCAL_LENGTH,
            fy=DEFAULT_FOCAL_LENGTH,
            cx=width / 2,
            cy=height / 2,
        ),
        ui=PerspectiveUi(
            depth_pro_fx_checkbox,
            fx_slider,
            fy_slider,
            camera_frustum,
            point_cloud,
        ),
    )

    def set_needs_refresh() -> None:
        logging.debug(f'Scheduling refresh for {perspective.name}')
        perspective.needs_refresh = True

    depth_pro_fx_checkbox.on_update(lambda _: set_needs_refresh())

    # Don't refresh if slider has moved but the checkbox DepthPro FOV is set.
    fx_slider.on_update(
        lambda _: (None if depth_pro_fx_checkbox.value else set_needs_refresh())
    )
    fy_slider.on_update(
        lambda _: (None if depth_pro_fx_checkbox.value else set_needs_refresh())
    )

    return perspective
