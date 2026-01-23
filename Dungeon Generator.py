"""
Dungeon Map Generator
"""

import numpy as np
from numpy import uint8
from numpy.typing import NDArray
from random import randint as rand
from os import system as sys
from Constants import *

def init_tilemap(size: int = 20):
    """
    Parameters
    ----------
    size : int
        Size of tilemap.

    Returns
    -------
    tilemap : NDArray[uint8]
        2D array of given size.
    """
    return np.zeros((size,size), uint8)

def room_fill(tilemap: NDArray[uint8]):
    """
    Parameters
    ----------
    tilemap : NDArray[uint8]
        Empty 2D array.

    Returns
    -------
    tilemap : NDArray[uint8]
        2D array with placed rooms of random size.
    """
    mid = DUNGEON_SIZE//2
    for _ in range(BOX_COUNT):
        y_s, y_e = rand(1, mid - 1), rand(mid + 2, DUNGEON_SIZE - 2)
        x_s, x_e = rand(1, mid - 1), rand(mid + 2, DUNGEON_SIZE - 2)
        tilemap[y_s:y_e, x_s:x_e] = ROOM
    return tilemap

def main():
    """
    Local Handler for Dungeon Generation
    ------------------------------------
    Only for use when running program from file\n
    Visualizer will be placed here once made
    """
    tilemap = init_tilemap()
    tilemap = room_fill(tilemap)
    print(tilemap)
    #Room Erosion/Filling Pass
    #Room Connector
    #Room Clearing Pass
    #Room Extension Pass
    #Tilemap Trim
    return

if __name__ == "__main__":
    sys("cls")
    main()