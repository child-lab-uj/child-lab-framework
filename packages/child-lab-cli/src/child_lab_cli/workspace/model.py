from dataclasses import dataclass
from pathlib import Path
from typing import Self

from serde.json import from_json
from transformation_buffer.buffer import Buffer


@dataclass(slots=True)
class Video:
    name: str
    location: Path


@dataclass(frozen=True, slots=True)
class NonCalibratedVideo:
    name: str
    location: Path


@dataclass(frozen=True, slots=True)
class CalibratedVideo:
    name: str
    location: Path
    calibration: Path


class WorkspaceModelError(Exception): ...


@dataclass(frozen=True, slots=True)
class Workspace:
    """
    Describes locations of the workspace components.

    Attributes
    ---
    name: str
        Name of the workspace.

    root: Path
        Path to the root directory of the workspace.

    input: list[Input]
        Directory with input videos to be processed.

    output: Path | None
        Directory for storing the resulsts of the analysis.

    archive: Path | None
        Directory for storing the results of the analysis and the input data
        from all the other workspace directories.

    calibration: Path | None
        Directory containing the serialized camera intrinsics for each camera.

    transformation: Path | None
        Directory containing the serialized transformation models.
    """

    name: str
    root: Path
    input: Path
    output: Path
    archive: Path
    calibration: Path
    transformation: Path

    def videos(self) -> list[Video]:
        return [Video(path.stem, path) for path in self.input.glob('*')]

    def calibrated_videos(self) -> list[CalibratedVideo]:
        return [
            CalibratedVideo(path.stem, path, calibration)
            for path in self.input.glob('*')
            if (calibration := self.calibration / f'{path.stem}.yml').is_file()
        ]

    def non_calibrated_videos(self) -> list[NonCalibratedVideo]:
        return [
            NonCalibratedVideo(path.stem, path)
            for path in self.input.glob('*')
            if not (self.calibration / path.name).is_file()
        ]

    def transformation_buffer(self) -> Buffer[object] | None:
        location = self.transformation / 'buffer.json'

        try:
            return from_json(Buffer, location.read_text())
        except Exception:
            return None

    @classmethod
    def in_directory(cls, root: Path) -> Self:
        """
        Constructs a new workspace with `root` root directory.

        Parameters
        ---
        root: Path
            Root directory of the workspace.

        Raises
        ---
        WorkspaceModelError
            If `root` is not an existing directory.
        """

        created_directories: list[Path] = []

        def maybe_create(path: Path) -> Path:
            if path.is_dir():
                return path

            if path.exists():
                raise WorkspaceModelError(f'{path} should be a directory.')

            path.mkdir()

            nonlocal created_directories
            created_directories.append(path)

            return path

        try:
            maybe_create(root)

            return cls(
                root.stem,
                root,
                maybe_create(root / 'input'),
                maybe_create(root / 'output'),
                maybe_create(root / 'archive'),
                maybe_create(root / 'calibration'),
                maybe_create(root / 'transformation'),
            )

        except Exception as exception:
            for path in reversed(created_directories):
                path.rmdir()

            raise exception
