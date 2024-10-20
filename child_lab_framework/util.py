import os
import pathlib


class CLFException(Exception):
    pass


WIDGETS_DIR = pathlib.Path(os.path.abspath(__file__)).parent.parent / 'widget'
MODELS_DIR = pathlib.Path(os.path.abspath(__file__)).parent.parent / 'model'
DEV_DIR = pathlib.Path(os.path.abspath(__file__)).parent.parent / 'dev'
