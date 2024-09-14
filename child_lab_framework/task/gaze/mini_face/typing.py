from typing import Protocol

from ....core.video import Frame

# NOTE: a temporary pseudo-type-stubs for dev/gaze-tracking/**/GazeTracking.so
# TODO: delete this module after issue #16 gets solved


type Vector3D = tuple[float, float, float]


class Gaze(Protocol):
    @property
    def eye1(self) -> Vector3D: ...

    @property
    def eye2(self) -> Vector3D: ...

    @property
    def direction1(self) -> Vector3D: ...

    @property
    def direction2(self) -> Vector3D: ...

    @property
    def angle(self) -> tuple[float, float]: ...


class GazeExtractor(Protocol):
    def set_camera_calibration(self, fx: float, fy: float, cx: float, cy: float) -> None: ...
    def detect_faces(self, frame: Frame, timestamp: float, region: tuple[float, float, float, float]) -> Gaze | None: ...
