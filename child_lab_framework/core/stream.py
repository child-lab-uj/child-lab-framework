from functools import wraps


class InvalidArgumentException(Exception): ...


def autostart(method):
    @wraps(method)
    def inner(self):
        # TODO: implement properly for async generators
        gen = method(self)
        return gen

    return inner
