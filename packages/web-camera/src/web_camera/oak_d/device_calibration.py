from dataclasses import dataclass
from operator import itemgetter
from typing import Self

import depthai as dai
from serde import field, json, serde


@serde
class Vector:
    x: float
    y: float
    z: float


@serde
class Extrinsics:
    rotation_matrix: list[list[float]] = field(rename='rotationMatrix')
    to_camera_socket: int = field(rename='toCameraSocket')
    spec_translation: Vector = field(rename='specTranslation')
    translation: Vector


@serde
class CameraData:
    camera_type: int = field(rename='cameraType')
    distortion: list[float] = field(rename='distortionCoeff')
    extrinsics: Extrinsics
    intrinsics: list[list[float]] = field(rename='intrinsicMatrix')
    lens_position: int = field(rename='lensPosition')
    spec_hfov_degree: float = field(rename='specHfovDeg')
    height: int
    width: int


@dataclass
class DeviceCalibration:
    color: CameraData
    left: CameraData
    right: CameraData

    @classmethod
    def parse(cls, calibration_handler: dai.CalibrationHandler) -> Self:
        camera_data = calibration_handler.eepromToJson()['cameraData']
        camera_data.sort(key=itemgetter(0))

        match camera_data:
            case [[0, dict(color_data)], [1, dict(left_data)], [2, dict(right_data)]]:
                color = json.from_dict(CameraData, color_data)
                left = json.from_dict(CameraData, left_data)
                right = json.from_dict(CameraData, right_data)

            case other:
                raise RuntimeError(f'Failed to parse cameraData: {other}')

        return cls(color, left, right)
