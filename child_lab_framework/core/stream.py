from functools import wraps
from ..typing.stream import Fiber


def autostart(method):
    @wraps(method)
    def inner(self):
        gen = method(self)
        next(gen)
        return gen

    return inner


def nones() -> Fiber[None, None]:
    while True:
        yield None
