"""
**Helper Functions used in both Generators**

Imported by: Dungeon_Generator, Room_Generator
"""

import numpy as np
from numpy import uint8
from numpy.typing import NDArray as array

def init_tilemap(height: int, width: int | None = None) -> array[uint8]:
    """
    Initializes an empty 2D tilemap filled with zeros.

    Parameters
    ----------
    height : int
        Height of the tilemap in tiles.
    width : int | None, optional
        Width of the tilemap in tiles. If not provided, width defaults
        to the same value as height, producing a square tilemap.

    Returns
    -------
    tilemap : NDArray[uint8]
        2D array of the given dimensions, initialized to all zeros.

    Notes
    -----
    - If width is None or 0, it is replaced with height via the
      expression `width = width or height`.
    - The array is initialized with numpy.zeros() using uint8 dtype,
      meaning each tile starts as 0 (inactive).
    """
    width = width or height
    tilemap = np.zeros((height,width), dtype = uint8)
    return tilemap