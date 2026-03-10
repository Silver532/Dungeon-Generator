"""
**Helper Functions used in both Generators**
"""

import numpy as np
from numpy import uint8
from numpy.typing import NDArray as array
from collections.abc import Collection

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

def get_direction_strings(value: int) -> tuple[str, ...]:
    """
    Converts an integer tile value into a tuple of direction strings
    based on its lower 4 bits.

    Parameters
    ----------
    value : int
        Bitmask value to extract directions from. The lower 4 bits
        are each mapped to an orthogonal direction:
            - Bit 0 (value 1) : North
            - Bit 1 (value 2) : East
            - Bit 2 (value 4) : South
            - Bit 3 (value 8) : West

    Returns
    -------
    dirs : tuple[str, ...]
        Tuple of direction strings corresponding to the set bits in
        the lower 4 bits of value. May be empty if no bits are set.

    Notes
    -----
    - Only the lower 4 bits of value are examined via `value & 0b01111`,
      so bit 4 (the active tile flag) and above are ignored.
    - The returned tuple preserves direction order: North, East, South,
      West, regardless of the input value.
    """
    bits = value & 0b01111
    directions = ('North','East','South','West')
    return tuple(direction for i, direction in enumerate(directions) if bits & (1 << i))

def adj_map(tilemap: array[uint8], neighbor_map:array[uint8] | None = None, iso:bool=True, target:Collection[int] | None = None) -> array[uint8]:
    """
    Calculates an orthogonal adjacency map for the given tilemap.

    Parameters
    ----------
    tilemap : NDArray[uint8]
        2D array with rooms placed.
    neighbor_map : NDArray[uint8] | None, optional
        2D array to write neighbor counts into. If not provided, a
        zero-filled array of the same shape as tilemap is created.
    iso : bool, optional
        If True, trims the result to only count neighbors for tiles
        that are active in the tilemap (value == 1). Defaults to True.
    target : set[int] | None, optional
        If provided, only tiles whose values are in this set are
        considered active when counting neighbors. If not provided,
        any non-zero tile is considered active.

    Returns
    -------
    neighbor_map : NDArray[uint8]
        2D array of the same dimensions as tilemap, where each value
        represents how many orthogonal neighbors the tile has.

    Notes
    -----
    - The active tile mask is built from either the target set (if
      provided) or any non-zero tile value.
    - Neighbor counts are accumulated using array slicing across all
      four orthogonal directions in two passes:
        - Vertical pass   : counts neighbors above and below
        - Horizontal pass : counts neighbors to the left and right
    - If iso is True, the neighbor map is masked by tiles with value
      exactly equal to 1, zeroing out counts for all other tiles.
    - An assertion is included to satisfy static type checkers that
      neighbor_map is non-None before returning.
    - Note: the original docstring listed the return type as 'tilemap'
      which has been corrected to 'neighbor_map'.
    """
    h, w = tilemap.shape
    if target is not None: mask = np.isin(tilemap, tuple(target))
    else: mask = tilemap != 0
    mask = mask.astype(uint8)
    if neighbor_map is None: neighbor_map = np.zeros_like(tilemap, dtype = uint8)
    else: neighbor_map.fill(0)

    neighbor_map[1:h-1, :] = mask[0:h-2, :] + mask[2:h, :]
    neighbor_map[:, 1:w-1] += mask[:, 0:w-2] + mask[:, 2:w]
    if iso: neighbor_map *= (tilemap == 1)
    assert neighbor_map is not None
    return neighbor_map