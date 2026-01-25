"""
Dungeon Map Generator
"""

import numpy as np
from numpy import uint8
from numpy.typing import NDArray
from random import randint as rand
from os import system as sys
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
        room_dims = 16 - (y_e-y_s)
        x_s = rand(1, mid - 1)
        x_e = min(x_s + room_dims + 1, DUNGEON_SIZE - 4)+2
        tilemap[y_s:y_e, x_s:x_e] = TEMP
    return tilemap

def adj_map(tilemap: NDArray[uint8]):
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
    orth_maps = []
    for i in [-1,1]:
        for j in [0,1]:
            orth_maps.append(np.roll(tilemap, i, j))
    neighbor_map = np.sum(orth_maps, axis = 0)
    neighbor_map *= tilemap
    return neighbor_map

def room_eroder(tilemap: NDArray[uint8]):
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
    neighbor_map = adj_map(tilemap)

    #print(f"There are {((tilemap == TEMP)).sum()} Rooms")   #DEBUG
    #print(neighbor_map,end="\n\n") #DEBUG
    #print(tilemap, end = "\n\n")   #DEBUG
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
    tilemap = room_eroder(tilemap)
    tilemap *= 16
    #Room Connector
    #Room Clearing Pass
    #Room Extension Pass
    #Tilemap Trim
    print(tilemap) #DEBUG
    return

if __name__ == "__main__":
    from time import perf_counter_ns as clock
    start_time = clock()
    sys("cls")
    main()
    end_time = clock()
    delta_time = (end_time - start_time)
    print(f"Program ran in {delta_time/1000000} milliseconds")