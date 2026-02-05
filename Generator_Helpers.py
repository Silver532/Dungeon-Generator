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

def _main() -> None:
    return

if __name__ == "__main__":
    _main()