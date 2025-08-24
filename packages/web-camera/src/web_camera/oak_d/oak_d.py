import logging
from dataclasses import dataclass
from typing import Final, cast

import depthai as dai
import kornia
import torch
from jaxtyping import Float32, UInt8
from torchvision.transforms import Resize
from video_io import Calibration

from .device_calibration import DeviceCalibration

__all__ = ['Metadata', 'CameraProperties', 'Reader']


@dataclass
class Metadata:
    fps: float
    height: int
    width: int


@dataclass
class CameraProperties:
    color_camera_calibration: Calibration
    stereo_camera_resolution: tuple[int, int]
    """(height, width)"""


class Reader:
    device: Final[torch.device]
    metadata: Final[Metadata]

    __camera: dai.Device
    __pipeline: dai.Pipeline
    __color_queue: dai.MessageQueue
    __depth_queue: dai.MessageQueue

    __depth_transformation: Resize

    def __init__(
        self,
        camera_name_or_id: str,
        fps: float = 60.0,
        width: int = 1920,
        height: int = 1080,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        camera = dai.Device(dai.DeviceInfo(deviceIdOrName=camera_name_or_id))
        pipeline = dai.Pipeline(camera)

        logging.debug(f'Connected to the camera at {camera_name_or_id}')

        color_cam = get_camera(pipeline, dai.CameraBoardSocket.CAM_A)
        left_cam = get_camera(pipeline, dai.CameraBoardSocket.CAM_B)
        right_cam = get_camera(pipeline, dai.CameraBoardSocket.CAM_C)

        # NOTE: Stereo has an arbitrary config.
        # Think about introducing a config structure or presets if more flexibility is required in the future.
        stereo = get_stereo_depth(pipeline)
        stereo.setRectification(True)
        stereo.setExtendedDisparity(True)
        stereo.setLeftRightCheck(True)
        # stereo.setDepthAlign(dai.CameraBoardSocket.CENTER)  # Crashes due to a mismatch of the color image size

        left_cam.requestFullResolutionOutput(
            type=dai.ImgFrame.Type.NV12,
            fps=fps,
            useHighestResolution=True,
        ).link(stereo.left)

        right_cam.requestFullResolutionOutput(
            type=dai.ImgFrame.Type.NV12,
            fps=fps,
            useHighestResolution=True,
        ).link(stereo.right)

        color_queue = color_cam.requestOutput(
            size=(width, height),
            type=dai.ImgFrame.Type.NV12,
            fps=fps,
            enableUndistortion=True,
        ).createOutputQueue(maxSize=4096)

        depth_queue = stereo.depth.createOutputQueue(maxSize=4096)

        self.device = device
        self.metadata = Metadata(fps, height, width)
        self.__camera = camera
        self.__pipeline = pipeline
        self.__color_queue = color_queue
        self.__depth_queue = depth_queue
        self.__depth_transformation = Resize((height, width))

        self.__pipeline.start()
        logging.debug('Started the DepthAI pipeline')

    def runtime_properties(self) -> CameraProperties:
        """
        Retrieves the non-persistent, runtime specification of the hardware.
        Warning: This is an expensive operation and may take a long time.
        """

        calibration = self.__camera.getCalibration()
        device_calibration = DeviceCalibration.parse(calibration)

        [[fx, _, cx], [_, fy, cy], *_] = calibration.getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A,
            resizeWidth=self.metadata.width,
            resizeHeight=self.metadata.height,
        )
        color_camera_calibration = Calibration(
            focal_length=(fx, fy),
            optical_center=(cx, cy),
            distortion=(0.0, 0.0, 0.0, 0.0, 0.0),
        )

        stereo_camera_resolution = (
            device_calibration.left.height,
            device_calibration.left.width,
        )

        return CameraProperties(color_camera_calibration, stereo_camera_resolution)

    def read(
        self,
    ) -> tuple[
        UInt8[torch.Tensor, '3 height width'],
        Float32[torch.Tensor, 'height width'],
    ]:
        color_message = self.__color_queue.get()
        depth_message = self.__depth_queue.get()

        assert isinstance(color_message, dai.ImgFrame)
        assert isinstance(depth_message, dai.ImgFrame)

        bgr = torch.from_numpy(color_message.getCvFrame())
        bgr = bgr.permute((2, 0, 1))
        rgb = kornia.color.bgr_to_rgb(bgr)
        rgb = rgb.to(self.device)

        depth = torch.from_numpy(depth_message.getFrame())
        depth = depth.to(self.device, torch.float32)

        return rgb, depth


def get_camera(pipeline: dai.Pipeline, socket: dai.CameraBoardSocket) -> dai.node.Camera:
    camera = cast(dai.node.Camera, pipeline.create(dai.node.Camera))
    return camera.build(socket)


def get_stereo_depth(pipeline: dai.Pipeline) -> dai.node.StereoDepth:
    return cast(dai.node.StereoDepth, pipeline.create(dai.node.StereoDepth))
