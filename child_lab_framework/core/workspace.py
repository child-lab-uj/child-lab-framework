from dataclasses import dataclass
from pathlib import Path
from typing import Self

from .calibration import Calibration
from .file import load
from .transformation import Buffer
from .video import SUPPORTED_SUFFIXES, Input


@dataclass(frozen=True)
class Workspace:
    """
    Represents the workspace, providing information about the input files
    with corresponding calibrations and the transformation buffer.

    A directory is considered a workspace if it contains subdirectories named
    `input`, `output`, `calibration`, `transformation` and `archive`.

    Attributes
    ---
    name: str
        Name of the workspace.

    root: Path
        Path to the root directory of the workspace.

    inputs: list[Input]
        Information of the input videos in the workspace.

    transformation_buffer: Buffer | None
        Transformation buffer describing camera poses used in the workspace, if available.
    """

    name: str
    root: Path
    inputs: list[Input]

    # The concrete type of the buffer does not matter for the Workspace,
    # it should be downcasted in order to be used.
    transformation_buffer: Buffer[object] | None

    @classmethod
    def load(cls, root: Path) -> Self:
        """
        Constructs a new workspace with `root` root directory.

        Parameters
        ---
        root: Path
            Root directory of the workspace.

        Raises
        ---
        RuntimeError
            If `root` is not a directory or does not contain some of the required components.
        """

        if not root.is_dir():
            raise RuntimeError(f'Cannot load workspace at {root} - not a directory')

        input_directory = root / 'input'
        if not input_directory.is_dir():
            raise RuntimeError(
                f'The workspace {root.stem} is missing an input directory at {input_directory}'
            )

        output_directory = root / 'output'
        if not output_directory.is_dir():
            raise RuntimeError(
                f'The workspace {root.stem} is missing an output directory at {output_directory}'
            )

        calibration_directory = root / 'calibration'
        if not calibration_directory.is_dir():
            raise RuntimeError(
                f'The workspace {root.stem} is missing a calibration directory at {calibration_directory}'
            )

        archive_directory = root / 'archive'
        if not archive_directory.is_dir():
            raise RuntimeError(
                f'The workspace {root.stem} is missing an archive directory at {archive_directory}'
            )

        transformation_directory = root / 'calibration'
        if not transformation_directory.is_dir():
            raise RuntimeError(
                f'The workspace {root.stem} is missing a transformation directory at {transformation_directory}'
            )

        transformation_buffer_file = root / 'transformation' / 'buffer.json'

        inputs = {
            file.stem: file
            for file in input_directory.iterdir()
            if file.suffix in SUPPORTED_SUFFIXES
        }

        calibrations = {
            name: load(Calibration, file)
            for file in calibration_directory.iterdir()
            if (name := file.stem) in inputs.keys()
        }

        buffer = (
            load(Buffer, transformation_buffer_file)
            if transformation_buffer_file.is_file()
            else None
        )

        return cls(
            root.stem,
            root,
            [
                Input(input_name, input_path, calibrations.get(input_name, None))
                for input_name, input_path in inputs.items()
            ],
            buffer,
        )
