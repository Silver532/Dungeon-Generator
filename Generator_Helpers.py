"""
**Helper Functions used in both Generators**
"""

import numpy as np
from numpy import uint8
from numpy.typing import NDArray as array

def init_tilemap(height: int, width: int | None = None) -> array[uint8]:
    """
    Initializes an empty tilemap

    Parameters
    ----------
    height : int
        height of tilemap.
    width : int | None = None
        width of tilemap.
        if no width given, use height instead

    Returns
    -------
    tilemap : NDArray[uint8]
        2D array of given size.
    """
    if width is not None: tilemap = np.zeros((height,width), uint8)
    else: tilemap = np.zeros((height,height), uint8)
    return tilemap

def get_directions(value: int) -> set[str]:
    """
    Converts integer tile value into room direction set

    Parameters
    ----------
    value: int
        Bitmask value to grab directions from\n
        1 = North\n
        2 = East\n
        4 = South\n
        8 = West

    Returns
    -------
    dirs: set[str]
        Set of direction strings
    """
    bits = value & 0b01111
    directions = ["North","East","South","West"]
    dirs = {directions[i] for i in range(4) if bits & (1 << i)}
    return dirs

def adj_map(tilemap: array[uint8], neighbor_map:array[uint8] | None = None, iso:bool=True, target:set[int] | None = None) -> array[uint8]:
    """
    Calculates adjacency maps for given targets

    Parameters
    ----------
    tilemap : NDArray[uint8]
        2D array with rooms placed.
    neighbor_map : NDArray[uint8] | None = None
        2D array for placing neighbor count in
        if no neighbor_map given, create zeroes_like
    iso : bool = True
        Trim values to only active tiles in the tilemap
    target : set[int] | None = None
        If provided, only these tile values are considered active

    Returns
    -------
    tilemap : NDArray[uint8]
        2D array counting how many neighbors each
        cell has in orthogonal directions.
    """
    h, w = tilemap.shape
    if target is not None: mask = np.isin(tilemap, tuple(target))
    else: mask = tilemap != 0
    mask = mask.astype(uint8)
    if neighbor_map is None: neighbor_map = np.zeros_like(tilemap)
    else: neighbor_map.fill(0)

    neighbor_map[1:h-1, :] = mask[0:h-2, :] + mask[2:h, :]
    neighbor_map[:, 1:w-1] += mask[:, 0:w-2] + mask[:, 2:w]
    if iso: neighbor_map *= (tilemap == 1)
    assert neighbor_map is not None
    return neighbor_map

def _main() -> None:
    """
    Generator_Helpers is a helper file, and does not run any code.
    """
    return

if __name__ == "__main__":
    _main()