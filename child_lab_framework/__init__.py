from child_lab_framework.hello import hello

import pathlib


class CLFException(Exception):
    pass


WIDGETS_DIR = pathlib.Path("../")
