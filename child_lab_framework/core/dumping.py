from typing import Protocol

import polars

# What describes my Result?
# 1. Numerical properties (representable as a vector, usually - floating-point, less common - integer, rare - enum, string)
# 2. Actors (usually ordered), enumerable with integer numbers

# How would the reader use the Result?
# 1. Igor needs to just read the 3D coordinates of a particular frame (henceforth, the timestamp is required)
# 2.
#
# What aspects of the Result does the reader need?
# 1. Which frame does it belong to
# 2. What properties does the n-th actor of the Result have


class Dumpable(Protocol):  # TODO: change the name to something more descriptive
    @property
    def data_frame(self) -> polars.DataFrame: ...
