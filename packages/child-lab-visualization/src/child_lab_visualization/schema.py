from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Frame:
    """
    Describes a single frame on the scene of a Viser server.
    Provides URIs of all components used by the visualization tools.
    """

    number: int

    def root(self) -> Path:
        return Path(f'/frame/{self.number}')

    def analysis(self) -> Path:
        return self.root() / 'analysis'

    def camera(self, name: str = 'origin') -> Path:
        return self.root() / 'camera' / name

    def frame_of_reference(self, name: str) -> Path:
        return self.root() / 'frame_of_reference' / name

    def point_cloud(self, frame_of_reference_name: str = 'origin') -> Path:
        return self.root() / 'point_cloud' / frame_of_reference_name

    def bounding_box(self, name: str, frame_of_reference_name: str = 'origin') -> Path:
        return self.analysis() / 'bounding_box' / frame_of_reference_name / name

    def pose(self, name: str, frame_of_reference_name: str = 'origin') -> Path:
        return self.analysis() / 'pose' / frame_of_reference_name / name

    def gaze(self, name: str, frame_of_reference_name: str = 'origin') -> Path:
        return self.analysis() / 'gaze' / frame_of_reference_name / name
