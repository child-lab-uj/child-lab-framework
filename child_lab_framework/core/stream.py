from functools import wraps


def autostart(method):
    @wraps(method)
    def inner(self):
        # TODO: implement properly for async generators
        gen = method(self)
        return gen

    return inner
