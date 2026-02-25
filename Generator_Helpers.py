"""
**Helper Functions for Generators**
"""

import numpy as np
from numpy import uint8
from numpy.typing import NDArray as array

def init_tilemap(height: int, width: int = 0) -> array[uint8]:
    """
    Parameters
    ----------
    height : int
        height of tilemap.
    width : int
        width of tilemap.

    Returns
    -------
    tilemap : NDArray[uint8]
        2D array of given size.
    """
    if width: tilemap = np.zeros((height,width), uint8)
    else: tilemap = np.zeros((height,height), uint8)
    return tilemap

def get_directions(value: int) -> set[str]:
    """
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

def adj_map(tilemap: array[uint8], neighbor_map:array[uint8] | None = None, iso:bool=True, target:set[int] | None=None) -> array[uint8]:
    """
    Parameters
    ----------
    tilemap : NDArray[uint8]
        2D array with rooms placed.
    neighbor_map : NDArray[uint8]
        2D array for placing neighbor count in
    iso : bool
        Trim values to only active tiles in the tilemap
    target : set[int] | None
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

    neighbor_map[1:h-1, :] = tilemap[0:h-2, :] + tilemap[2:h, :]
    neighbor_map[:, 1:w-1] += tilemap[:, 0:w-2] + tilemap[:, 2:w]
    if iso: neighbor_map *= (tilemap == 1)
    assert neighbor_map is not None
    return neighbor_map

def _main() -> None:
    return

if __name__ == "__main__":
    _main()