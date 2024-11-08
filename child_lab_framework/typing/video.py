from typing import Literal

import numpy as np

type Frame = np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.uint8]]
