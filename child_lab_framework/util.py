import pathlib
import os


class CLFException(Exception):
    pass


WIDGETS_DIR = pathlib.Path(os.path.abspath(__file__)).parent / "../widget"
MODELS_DIR = pathlib.Path(os.path.abspath(__file__)).parent / "../model"
