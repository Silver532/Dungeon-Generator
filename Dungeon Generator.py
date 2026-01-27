"""
Dungeon Map Generator
"""

import numpy as np
from numpy import uint8
from numpy.typing import NDArray as array
from random import randint as rand
from Constants import *

def init_tilemap(size: int = DUNGEON_SIZE):
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

def room_fill(tilemap: array[uint8]):
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
        room_dims = 16 - (y_e-y_s)
        x_s = rand(1, mid - 1)
        x_e = min(x_s + room_dims + 3, DUNGEON_SIZE - 4)+2
        tilemap[y_s:y_e, x_s:x_e] = TEMP
    return tilemap

def adj_map(tilemap: array[uint8], neighbor_map: array[uint8], iso: bool = True):
    """
    Parameters
    ----------
    tilemap : NDArray[uint8]
        2D array with rooms placed.

    Returns
    -------
    tilemap : NDArray[int]
        2D array counting how many neighbors each cell has in orthogonal directions.
    """
    neighbor_map.fill(0)

    neighbor_map[1:19, :] = tilemap[0:18, :] + tilemap[2:20, :]
    neighbor_map[:, 1:19] += tilemap[:, 0:18] + tilemap[:, 2:20]
    if iso: neighbor_map *= tilemap
    return neighbor_map

def room_eroder(tilemap: array[uint8]):
    """
    Parameters
    ----------
    tilemap : NDArray[uint8]
        2D array with rooms placed.

    Returns
    -------
    tilemap : NDArray[int]
        2D array with room edges eroded for smoother generation.
    """
    zeroes = np.zeros_like(tilemap, uint8)
    for _ in range(ERODE_COUNT):
        neighbor_map = adj_map(tilemap, zeroes)
        for index, i in np.ndenumerate(neighbor_map==2):
            if i & rand(0,1): tilemap[index] = WALL
        for index, i in np.ndenumerate(neighbor_map==3):
            if i & (rand(0,9)==0): tilemap[index] = WALL
    neighbor_map = adj_map(tilemap, zeroes)
    for index, i in np.ndenumerate(neighbor_map==0):
        if i: tilemap[index] = WALL
    return tilemap

def main():
    """
    Local Handler for Dungeon Generation
    ------------------------------------
    Only for use when running program from file\n
    Visualizer will be placed here once made
    """
    from time import perf_counter_ns as clock
    print("\033c", end="")
    start_time = clock()

    tilemap = init_tilemap()
    tilemap = room_fill(tilemap)
    tilemap = room_eroder(tilemap)
    tilemap *= 16
    #Room Connector
    #Room Clearing Pass
    #Room Extension Pass
    #Tilemap Trim
    print(tilemap) #DEBUG

    end_time = clock()
    delta_time = (end_time - start_time)
    print(f"Program ran in {delta_time/1000000} milliseconds")
    return

if __name__ == "__main__":
    main()