from pathlib import Path

from more_itertools import first_true


def __workspace_root() -> Path:
    for parent in Path(__file__).parents:
        content = parent.glob('*')

        if first_true(content, None, lambda file: file.name == 'uv.lock') is not None:
            return parent

    assert False, 'unreachable'


WORKSPACE_ROOT = __workspace_root()
DEVELOPMENT_DIRECTORY = WORKSPACE_ROOT / 'development'
TEST_DATA_DIRECTORY = DEVELOPMENT_DIRECTORY / 'test_data'
