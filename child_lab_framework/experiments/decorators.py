import inspect
import typing
from collections.abc import Callable
from functools import wraps

ENDPOINT_TYPES: dict[str, type] = {
    'camera_info': tuple[int, int],
    'transformation': tuple[float, float, float],
}


def link(subscribe: tuple[str, ...], publish: tuple[str, ...]) -> Callable:
    subscribed_types = tuple(map(ENDPOINT_TYPES.__getitem__, subscribe))
    # published_types = tuple(map(ENDPOINT_TYPES.__getitem__, publish))

    def inner(method: Callable) -> Callable:
        arg_names = inspect.getfullargspec(method)[0]
        arg_names.remove('self')

        declared_arg_types = typing.get_type_hints(method, include_extras=True)
        return_type = declared_arg_types.pop('return')  # pyright: ignore

        print(f'{subscribed_types = }')
        print(f'{arg_names = }')
        print(f'{declared_arg_types = }')

        assert len(arg_names) == len(subscribed_types)

        for name, expected_type in zip(arg_names, subscribed_types):
            declared_type = declared_arg_types[name]
            assert (
                declared_type == expected_type
            ), f'Expected {expected_type}, got {declared_type}'

        @wraps(method)
        def wrapper(self, *args):
            return method(self, *args)

        return wrapper

    return inner


class Component:
    @link(subscribe=('camera_info',), publish=('transformation',))
    def __process(self, arg: tuple[int, int]) -> float: ...


c = Component()
