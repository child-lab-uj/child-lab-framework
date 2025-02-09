import os
import tomllib
from pathlib import Path


def run_mypy_for_workspace_members() -> None:
    manifest_path = Path(__file__).parent / 'pyproject.toml'
    manifest = tomllib.loads(manifest_path.read_text())

    workspace_members: list[str] = manifest['tool']['uv']['workspace']['members']

    for member in map(Path, workspace_members):
        member_concrete_paths = member.parent.glob(member.name)

        for member_path in member_concrete_paths:
            status = os.system(f'uv run mypy {member_path} --config-file pyproject.toml')

            if status != 0:
                break


if __name__ == '__main__':
    run_mypy_for_workspace_members()
