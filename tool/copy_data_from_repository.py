import shutil
import sys
from pathlib import Path

from colorama import Fore

ALLOWED_EXTENSIONS = ['.avi', '.mp4']


NAME_TRANSLATION: dict[str, str] = {
    'Kamera 1': 'window_left',
    'Kamera 2': 'window_right',
    'Kamera 3': 'wall_left',
    'Kamera 4': 'wall_right',
    'Kamera 5': 'ceiling',
}


def main(source: Path, destination: Path) -> None:
    sources = [path for path in source.rglob('*') if path.suffix in ALLOWED_EXTENSIONS]
    destinations = [
        destination
        / (
            NAME_TRANSLATION[str(path.parent.name)]
            + (
                ('_' + name[-1])
                if (name := path.name.removesuffix(path.suffix))[-1].isalpha()
                else ''
            )
            + path.suffix
        )
        for path in sources
    ]

    for src, dest in zip(sources, destinations):
        print(
            'Copying '
            + Fore.GREEN
            + str(src)
            + Fore.RESET
            + ' to '
            + Fore.BLUE
            + str(dest)
            + Fore.RESET
        )

        shutil.copy(src, dest)


if __name__ == '__main__':
    _, source, destination, *_ = sys.argv

    source = Path(source)
    destination = Path(destination)

    assert source.is_dir()
    assert destination.is_dir()

    main(source, destination)
