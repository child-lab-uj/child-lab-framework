import asyncio
from functools import wraps

from ..typing.stream import Fiber


def autostart(method):
    @wraps(method)
    def inner(self):
        # TODO: implement properly for async generators
        gen = method(self)
        return gen

    return inner
