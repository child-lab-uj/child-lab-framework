from threading import Lock
from typing import Protocol

import colorama


class Representable(Protocol):
    def __repr__(self) -> str: ...


class Logger:
    LOCK = Lock()

    @staticmethod
    def info(*args: Representable, title: str = 'Info') -> None:
        title = colorama.Fore.GREEN + f'[{title}]:' + colorama.Fore.RESET

        with Logger.LOCK:
            print(title, *args, flush=True)

    @staticmethod
    def error(*args: Representable, title: str = 'Error') -> None:
        title = colorama.Fore.RED + f'[{title}]:' + colorama.Fore.RESET

        with Logger.LOCK:
            print(title, *args, flush=True)
