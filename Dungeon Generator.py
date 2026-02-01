"""
**Dungeon Map Generator**
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy import uint8
from numpy.typing import NDArray as array
from random import randint as rand
from random import random
from random import sample
from matplotlib import rcParams
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.axes import Axes

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

def adj_map(tilemap: array[uint8], neighbor_map:array[uint8], iso:bool=True):
    """
    Parameters
    ----------
    tilemap : NDArray[uint8]
        2D array with rooms placed.
    neighbor_map : NDArray[uint8]
        2D array for placing neighbor count in
    iso : bool
        Trim values to only active tiles in the tilemap

    Returns
    -------
    tilemap : NDArray[int]
        2D array counting how many neighbors each
        cell has in orthogonal directions.
    """
    h, w = tilemap.shape
    neighbor_map.fill(0)

    neighbor_map[1:h-1, :] = tilemap[0:h-2, :] + tilemap[2:h, :]
    neighbor_map[:, 1:w-1] += tilemap[:, 0:w-2] + tilemap[:, 2:w]
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

def tilemap_trim(tilemap: array[uint8]):
    active_rows = np.any(tilemap != 0, axis=1)
    active_cols = np.any(tilemap != 0, axis=0)
    trimmed_tilemap = tilemap[np.ix_(active_rows, active_cols)]
    return trimmed_tilemap

def get_directions(value: int):
    bits = value & 0b01111
    directions = ["North","East","South","West"]
    dirs = [directions[i] for i in range(4) if bits & (1 << i)]
    return dirs

def get_possible_connections(tilemap: array[uint8]):
    up    = tilemap[:-2, 1:-1] != 0
    right = tilemap[1:-1, 2:] != 0
    down  = tilemap[2:, 1:-1] != 0
    left  = tilemap[1:-1, :-2] != 0

    connections = (
        up.astype(np.uint8)
        | (right.astype(np.uint8) << 1)
        | (down.astype(np.uint8) << 2)
        | (left.astype(np.uint8) << 3)
    )
    connections *= tilemap[1:-1, 1:-1] != 0
    return connections

def room_random():
    r = random()
    if r < 0.60:    return 1
    elif r < 0.85:  return 2
    else:           return 3

def room_connector(tilemap: array[uint8]):
    connection_map = get_possible_connections(tilemap)
    H, W = tilemap.shape
    dir_to_bit = {"North": 1 << 0, "East":  1 << 1, "South": 1 << 2, "West":  1 << 3}
    opposite = {"North": "South", "East":  "West", "South": "North", "West":  "East",}
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if tilemap[y, x] == 0: continue
            possible_dirs = connection_map[y - 1, x - 1]
            dir_list = get_directions(possible_dirs)
            connect_count = min(room_random(), len(dir_list))
            chosen_dirs = sample(dir_list, connect_count)
            for d in chosen_dirs:
                tilemap[y,x] |= dir_to_bit[d]
                match d:
                    case "North": ny, nx = y-1, x
                    case "South": ny, nx = y+1, x
                    case "East": ny, nx = y, x+1
                    case "West": ny, nx = y, x-1
                    case _: ny, nx = y,x

                tilemap[ny, nx] |= dir_to_bit[opposite[d]]
    return tilemap

def dungeon_map_generator():
    """
    Dungeon Map Generator
    ---------------------
    Importable Handler for Dungeon Map Generation
    """
    tilemap = init_tilemap()
    tilemap = room_fill(tilemap)
    tilemap = room_eroder(tilemap)
    tilemap *= 16
    tilemap = room_connector(tilemap)
    tilemap = tilemap_trim(tilemap)
    #Room Clearing Pass
    return tilemap

def _make_exit_map(tilemap: array[uint8]):
    """
    Local Subhandler for Dungeon Map visualizer
    """
    debug_map = np.zeros_like(tilemap, uint8)
    for index, val in np.ndenumerate(tilemap):
        debug_map[index] = np.bitwise_count(val)
    return debug_map

def _on_click(event, ax: Axes, tilemap: array[uint8]):
    if event.inaxes != ax: return
    col = int(event.xdata+0.5)
    row = int(event.ydata+0.5)
    if 0 <= row < tilemap.shape[0] and 0 <= col < tilemap.shape[1]:
        dirs = get_directions(tilemap[row,col])
        print(f"\033cTile Clicked: {row}, {col}\n"+
              f"Tile Value: {tilemap[row,col]}\n"+
              f"Exits: {", ".join(dirs)}")
    return

def _debug(tilemap: array[uint8]):
    """
    Local Handler for Debug Purposes
    --------------------------------
    Visualizer and debug info for dungeon map
    """
    #for index, i in np.ndenumerate(tilemap):       #DEBUG
    #    if i: tilemap[index] += choice([1,2,4,8])  #DEBUG

    debug_map = _make_exit_map(tilemap)

    colours = ["white", "black", "green", "blue", "red", "yellow"]
    
    cmap = ListedColormap(colours)
    norm = BoundaryNorm(range(len(colours)+1), cmap.N)
    
    rcParams["toolbar"]="None"
    fig, ax = plt.subplots(figsize = (5,5), dpi = 120)

    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title("DEBUG Window")
    
    rows, cols = debug_map.shape

    ax.imshow(debug_map,cmap=cmap,norm=norm,interpolation="nearest")
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)

    fig.canvas.mpl_connect("button_press_event",lambda event: _on_click(event, ax, tilemap))

    plt.show()
    return

def _main():
    """
    Local Handler for Dungeon Generation
    ------------------------------------
    Only for use when running program from file
    """
    from time import perf_counter_ns as clock
    print("\033c", end="")
    start_time = clock()

    tilemap = init_tilemap()
    tilemap = room_fill(tilemap)
    tilemap = room_eroder(tilemap)
    tilemap *= 16
    tilemap = room_connector(tilemap)
    tilemap = tilemap_trim(tilemap)
    #Room Clearing Pass

    end_time = clock()
    delta_time = (end_time - start_time)/1000000
    print(f"Program ran in {delta_time} milliseconds")
    _debug(tilemap)
    return

if __name__ == "__main__":
    _main()