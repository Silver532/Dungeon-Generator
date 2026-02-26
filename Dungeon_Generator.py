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
from matplotlib.backend_bases import Event, MouseEvent
from typing import Literal

from Constants import Dungeon_Generator_Constants as const
from Generator_Helpers import *

def room_fill(tilemap: array[uint8]) -> array[uint8]:
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
    mid = const.DUNGEON_SIZE//2
    for _ in range(const.BOX_COUNT):
        y_s, y_e = rand(1, mid - 1), rand(mid + 2, const.DUNGEON_SIZE - 2)
        room_dims = 16 - (y_e-y_s)
        x_s = rand(1, mid - 1)
        x_e = min(x_s + room_dims + 3, const.DUNGEON_SIZE - 4)+2
        tilemap[y_s:y_e, x_s:x_e] = const.TEMP
    return tilemap

def room_eroder(tilemap: array[uint8]) -> array[uint8]:
    """
    Parameters
    ----------
    tilemap : NDArray[uint8]
        2D array with rooms placed.

    Returns
    -------
    tilemap : NDArray[uint8]
        2D array with room edges eroded for smoother generation.
    """
    zeroes = np.zeros_like(tilemap, dtype=uint8)

    for _ in range(const.ERODE_COUNT):
        neighbor_map = adj_map(tilemap, zeroes)

        coords2 = np.argwhere(neighbor_map == 2)
        for y, x in coords2:
            if rand(0,1):
                tilemap[y, x] = const.NO_ROOM

        coords3 = np.argwhere(neighbor_map == 3)
        for y, x in coords3:
            if rand(0,9) == 0:
                tilemap[y, x] = const.NO_ROOM

    neighbor_map = adj_map(tilemap, zeroes)
    coords0 = np.argwhere(neighbor_map == 0)
    for y, x in coords0:
        tilemap[y, x] = const.NO_ROOM

    return tilemap

def get_possible_connections(tilemap: array[uint8]) -> array[uint8]:
    """
    Parameters
    ----------
    tilemap : NDArray[uint8]
        2D array containing active and inactive tiles.

    Returns
    -------
    tilemap : NDArray[uint8]
        2D array of same dimensions, containing number
        of active tiles adjacent to each tile.
    """
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

def room_random() -> Literal[1,2,3]:
    """
    Returns
    -------
    Literal[1, 2, 3]
        returns 1, 2 or 3 with a weighted random
    """
    r = random()
    if r < 0.55:    return 1
    elif r < 0.80:  return 2
    else:           return 3

def room_connector(tilemap: array[uint8]) -> array[uint8]:
    """
    Parameters
    ----------
    tilemap : NDArray[uint8]
        2D array with active rooms (bit 4).

    Returns
    -------
    tilemap : NDArray[uint8]
        2D array with rooms connected (bits 0-3).
    """
    connection_map = get_possible_connections(tilemap)
    H, W = tilemap.shape
    dir_to_bit = {"North": 1 << 0, "East":  1 << 1, "South": 1 << 2, "West":  1 << 3}
    opposite = {"North": "South", "East":  "West", "South": "North", "West":  "East",}
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if tilemap[y, x] == 0: continue
            possible_dirs = connection_map[y - 1, x - 1]
            dir_set = get_directions(possible_dirs)
            connect_count = min(room_random(), len(dir_set))
            chosen_dirs = sample(tuple(dir_set), connect_count)
            for d in chosen_dirs:
                tilemap[y,x] |= dir_to_bit[d]
                dy_dx = {"North": (-1,0), "South": (1,0), "East": (0,1), "West": (0,-1)}
                ny, nx = y + dy_dx[d][0], x + dy_dx[d][1]
                tilemap[ny, nx] |= dir_to_bit[opposite[d]]
    return tilemap

def tilemap_trim(tilemap: array[uint8]) -> array[uint8]:
    """
    Parameters
    ----------
    tilemap : NDArray[uint8]
        2D array with rooms in final positions.

    Returns
    -------
    tilemap : NDArray[uint8]
        2D array that contains only the bounding box of the room positions.
    """
    active_rows = np.any(tilemap != 0, axis=1)
    active_cols = np.any(tilemap != 0, axis=0)
    trimmed_tilemap = tilemap[np.ix_(active_rows, active_cols)]
    return trimmed_tilemap

def room_clear(tilemap: array[uint8]) -> array[uint8]:
    """
    Parameters
    ----------
    tilemap : NDArray[uint8]
        2D array with room connections.

    Returns
    -------
    tilemap : NDArray[unit8]
        2D array with unconnected rooms removed.
        Only affects groups of 2.
    """
    DIR_MAP = {17:(-1,0), 18:(0,1), 20:(1,0), 24:(0,-1)}
    ONE_EXIT_TILES = [17,18,20,24]
    for index, i in np.ndenumerate(tilemap):
        if i in ONE_EXIT_TILES:
            adj_tile = DIR_MAP[int(i)]
            adj_tile_val = tuple(a + b for a, b in zip(index, adj_tile))
            if tilemap[adj_tile_val] in ONE_EXIT_TILES:
                tilemap[index] = 0
                tilemap[adj_tile_val] = 0
    return tilemap

def dungeon_map_generator() -> array[uint8]:
    """
    Dungeon Map Generator
    ---------------------
    Importable Handler for Dungeon Map Generation
    """
    tilemap = init_tilemap(const.DUNGEON_SIZE)
    tilemap = room_fill(tilemap)
    tilemap = room_eroder(tilemap)
    tilemap *= 16
    tilemap = room_connector(tilemap)
    tilemap = tilemap_trim(tilemap)
    tilemap = room_clear(tilemap)
    return tilemap

#region DEBUG
def _make_exit_map(tilemap: array[uint8]) -> array[uint8]:
    """
    Local Subhandler for Dungeon Map visualizer
    Vectorized version using numpy.
    """
    debug_map = np.unpackbits(tilemap[:, :, np.newaxis], axis=-1).sum(axis=-1).astype(np.uint8)
    return debug_map

def _on_click(event: Event, ax: Axes, tilemap: array[uint8], time: float, room_count: int) -> None:
    """
    Local handler for debug on click event.
    """
    if not isinstance(event, MouseEvent): return
    if event.inaxes is ax and event.xdata is not None and event.ydata is not None:
        col = int(event.xdata+0.5)
        row = int(event.ydata+0.5)
        if 0 <= row < tilemap.shape[0] and 0 <= col < tilemap.shape[1]:
            dirs = get_directions(tilemap[row,col])
            print(f"\033cProgram ran in {time} milliseconds\n"+
                f"Tile Clicked: {row}, {col}\n"+
                f"Tile Value: {tilemap[row,col]}\n"+
                f"Exits: {", ".join(dirs)}")
    else:
        print(f"\033cProgram ran in {time} milliseconds\n"+
              f"Dungeon contains {room_count} rooms")
    return

def _debug(tilemap: array[uint8], time: float, room_count: int) -> None:
    """
    Local Handler for Debug Purposes
    --------------------------------
    Visualizer and debug info for dungeon map
    """
    debug_map = _make_exit_map(tilemap)

    colours = ["white", "black", "green", "blue", "red", "yellow"]
    
    cmap = ListedColormap(colours)
    norm = BoundaryNorm(range(len(colours)+1), cmap.N)
    
    rcParams["toolbar"]="None"
    # False positive from Matplotlib type stubs.
    # Many pyplot/Axes methods define **kwargs as Unknown, which triggers
    # reportUnknownMemberType under strict mode.
    # Argument and return types are otherwise fully resolved and type-safe.
    fig, ax = plt.subplots(figsize = (5,5), dpi = 120)                                          #type: ignore[reportUnknownMemberType]

    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title("DEBUG Window")
    
    rows, cols = debug_map.shape

    ax.imshow(debug_map,cmap=cmap,norm=norm,interpolation="nearest")                            #type: ignore[reportUnknownMemberType]
    ax.grid(which="minor", color="white", linewidth=0.5)                                        #type: ignore[reportUnknownMemberType]
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)  #type: ignore[reportUnknownMemberType]
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)                                         #type: ignore[reportUnknownMemberType]
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)                                         #type: ignore[reportUnknownMemberType]

    fig.canvas.mpl_connect("button_press_event",lambda event:
                           _on_click(event,ax,tilemap,time,room_count))

    plt.show()                                                                                  #type: ignore[reportUnknownMemberType]
    return

def _main() -> None:
    """
    Local Handler for Dungeon Generation
    ------------------------------------
    Only for use when running program from file
    """
    from time import perf_counter_ns as clock
    print("\033c", end="")
    start_time = clock()

    tilemap = dungeon_map_generator()

    end_time = clock()
    delta_time = (end_time - start_time)/1000000
    room_count = np.count_nonzero(tilemap)
    print(f"Program ran in {delta_time} milliseconds")
    print(f"Dungeon contains {room_count} rooms")
    _debug(tilemap, delta_time, room_count)
    return
#endregion

if __name__ == "__main__":
    _main()