"""
**Dungeon Map Generator**
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy import uint8
from numpy.typing import NDArray as array
from matplotlib import rcParams
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event, MouseEvent
from time import perf_counter_ns as clock
from enum import IntEnum
from random import Random

from Generator_Helpers import init_tilemap, adj_map, get_direction_strings
from Debug_Tools import timeit, arg_parser

class dirs():
    DIRS = ('North','East','South','West')
    DY_DX_LIST = ((-1,0),(0,1),(1,0),(0,-1))
    DIR_BITS = (1,2,4,8)
    OPPOSITE_BITS = (4,8,1,2)
    DIR_IDX = {d: i for i, d in enumerate(DIRS)}
    ONE_EXIT_TILES = {17, 18, 20, 24}
    DIR_OFFSETS = {17:(-1,0), 18:(0,1), 20:(1,0), 24:(0,-1)}
    MASK_TO_INDICES: tuple[tuple[int, ...], ...] = tuple(tuple(i for i in range(4) if mask & (1 << i)) for mask in range(16))

class const(IntEnum):
    """
    Constants for Dungeon_Generator file
    """
    DUNGEON_SIZE = 20
    MID = DUNGEON_SIZE//2
    BOX_COUNT = 3
    ERODE_COUNT = 5
    NORTH = 1
    EAST = 2
    SOUTH = 4
    WEST = 8
    TEMP = 1
    NO_ROOM = 0
    ROOM = 16

@timeit
def room_fill(tilemap: array[uint8], np_rng: np.random.Generator, rand_rng: Random) -> array[uint8]:
    """
    Places random rooms inside an array
    
    Parameters
    ----------
    tilemap : NDArray[uint8]
        Empty 2D array.

    Returns
    -------
    tilemap : NDArray[uint8]
        2D array with placed rooms of random size.
    """
    for _ in range(const.BOX_COUNT):
        y_s, y_e = np_rng.integers(1, const.MID), np_rng.integers(const.MID + 2, const.DUNGEON_SIZE - 1)
        room_dims = 16 - (y_e-y_s)
        x_s = np_rng.integers(1, const.MID)
        x_e = min(x_s + room_dims + 5, const.DUNGEON_SIZE - 2)
        tilemap[y_s:y_e, x_s:x_e] = const.TEMP
    return tilemap

@timeit
def room_eroder(tilemap: array[uint8], np_rng: np.random.Generator, rand_rng: Random) -> array[uint8]:
    """
    Erodes rooms inside array

    Parameters
    ----------
    tilemap : NDArray[uint8]
        2D array with rooms placed.

    Returns
    -------
    tilemap : NDArray[uint8]
        2D array with room edges eroded for smoother generation.
    """
    neighbor_map = np.zeros_like(tilemap, dtype=uint8)

    for _ in range(const.ERODE_COUNT):
        adj_map(tilemap, neighbor_map)

        mask2 = (neighbor_map == 2)
        tilemap[mask2 & (np_rng.random(mask2.shape, dtype = np.float32) < 0.5)] = const.NO_ROOM

        mask3 = (neighbor_map == 3)
        tilemap[mask3 & (np_rng.random(mask3.shape, dtype = np.float32) < 0.1)] = const.NO_ROOM

    adj_map(tilemap, neighbor_map)
    tilemap[neighbor_map == 0] = const.NO_ROOM
    return tilemap

@timeit
def get_possible_connections(tilemap: array[uint8]) -> array[uint8]:
    """
    Converts tilemap into map of adjacent tile count

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
    t = (tilemap != 0).astype(uint8)
    connections = (
    t[:-2, 1:-1]
    | (t[1:-1, 2:] << 1)
    | (t[2:, 1:-1] << 2)
    | (t[1:-1, :-2] << 3)
)
    connections *= t[1:-1, 1:-1]
    return connections

@timeit
def room_random(np_rng: np.random.Generator, count: int) -> array[uint8]:
    """
    Generate weighted random values for room connections

    Parameters
    ----------
    np_rng : np.random.Generator
        numpy random generator
    count : int
        number of random values to generate.

    Returns
    -------
    randoms : NDArray[uint8]
        Array with random values
    """
    r = np_rng.random(count, dtype = np.float32)

    randoms = np.ones(count, dtype = uint8)
    randoms[r >= 0.55] = 2
    randoms[r >= 0.80] = 3
    return randoms

@timeit
def room_connector(tilemap: array[uint8], np_rng: np.random.Generator, rand_rng: Random) -> array[uint8]:
    """
    Connects adjacent active rooms across given tilemap

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
    active_count = np.count_nonzero(tilemap != 0)
    connection_counts = room_random(np_rng, active_count)
    mask_lookup = dirs.MASK_TO_INDICES
    DIR_BITS = dirs.DIR_BITS
    DY_DX = dirs.DY_DX_LIST
    OPP_BITS = dirs.OPPOSITE_BITS
    randrange = rand_rng.randrange

    connection_index = 0

    for y in range(1, H - 1):
        row = tilemap[y]
        if not row.any():
            continue
        for x in range(1, W - 1):
            if row[x] == 0:
                continue

            mask: uint8 = connection_map[y - 1, x - 1] & 0b1111
            if mask == 0:
                continue

            indices = mask_lookup[mask]
            n = len(indices)

            connect_count = connection_counts[connection_index]
            connection_index += 1

            if connect_count >= n:
                chosen = indices
            elif connect_count == 1:
                chosen = [indices[randrange(n)]]
            else:
                chosen = rand_rng.sample(indices, connect_count)

            for i in chosen:
                row[x] |= DIR_BITS[i]
                dy, dx = DY_DX[i]
                ny = y + dy
                nx = x + dx
                tilemap[ny, nx] |= OPP_BITS[i]
    return tilemap

@timeit
def tilemap_trim(tilemap: array[uint8]) -> array[uint8]:
    """
    Reduces size of tilemap to smallest possible without removing active tiles

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

@timeit
def room_clear(tilemap: array[uint8]) -> array[uint8]:
    """
    Removes unconnected rooms from tilemap

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
    for index, val in np.ndenumerate(tilemap):
        if val in dirs.ONE_EXIT_TILES:
            dy, dx = dirs.DIR_OFFSETS[int(val)]
            ny, nx = index[0]+dy, index[1]+dx
            if tilemap[ny, nx] in dirs.ONE_EXIT_TILES:
                tilemap[index] = 0
                tilemap[ny, nx] = 0
    return tilemap

@timeit
def dungeon_map_generator(np_rng: np.random.Generator, rand_rng: Random) -> array[uint8]:
    """
    Handler function to create dungeon map\n
    tilemap is modified and returned each step

    Parameters
    ----------
    seed : int | None = None
        if provided, use this seed to provide rng

    Returns
    -------
    tilemap : NDArray[uint8]
        Dungeon map array
    """
    tilemap = init_tilemap(const.DUNGEON_SIZE)
    tilemap = room_fill(tilemap, np_rng, rand_rng)
    tilemap = room_eroder(tilemap, np_rng, rand_rng)
    tilemap *= 16
    tilemap = room_connector(tilemap, np_rng, rand_rng)
    tilemap = room_clear(tilemap)
    tilemap = tilemap_trim(tilemap)
    return tilemap

#region DEBUG
def _make_exit_map(tilemap: array[uint8]) -> array[uint8]:
    """
    Local function that converts tilemap into a display map

    Parameters
    ----------
    tilemap : NDArray[uint8]
        2D array to be converted

    Returns
    -------
    debug_map : NDArray[uint8]
        2D array converted to exit count map for display
    """
    debug_map = np.unpackbits(tilemap[:, :, np.newaxis], axis=-1).sum(axis=-1).astype(np.uint8)
    return debug_map

def _on_click(event: Event, ax: Axes, tilemap: array[uint8], room_count: int) -> None:
    """
    Local handler for debug click events

    Parameters
    ----------
    event : Event
        matplotlib click event
    ax : Axes
        matplotlib graph axes
    tilemap : NDArray[uint8]
        tilemap of the dungeon
    time : float
        number of ms it took for dungeon to generate
    room_count : int
        number of rooms in the dungeon
    """
    if not isinstance(event, MouseEvent): return
    if event.inaxes is ax and event.xdata is not None and event.ydata is not None:
        col = int(event.xdata+0.5)
        row = int(event.ydata+0.5)
        if 0 <= row < tilemap.shape[0] and 0 <= col < tilemap.shape[1]:
            dirs = get_direction_strings(tilemap[row,col])
            print(f"\033cTile Clicked: {row}, {col}\n"+
                f"Tile Value: {tilemap[row,col]}\n"+
                f"Exits: {", ".join(dirs)}")
    else:
        print(f"\033cDungeon contains {room_count} rooms")
    return

def _debug(tilemap: array[uint8], room_count: int) -> None:
    """
    Local handler for visualization and debugging

    Parameters
    ----------
    tilemap : NDArray[uint8]
        dungeon tilemap
    time : float
        number of ms it took for dungeon to generate
    room_count : int
        number of rooms in the dungeon  
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
    fig, ax = plt.subplots(figsize = (5,5), dpi = 120)                                          #pyright: ignore[reportUnknownMemberType]

    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title("DEBUG Window")
    
    rows, cols = debug_map.shape

    ax.imshow(debug_map,cmap=cmap,norm=norm,interpolation="nearest")                            #pyright: ignore[reportUnknownMemberType]
    ax.grid(which="minor", color="white", linewidth=0.5)                                        #pyright: ignore[reportUnknownMemberType]
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)  #pyright: ignore[reportUnknownMemberType]
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)                                         #pyright: ignore[reportUnknownMemberType]
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)                                         #pyright: ignore[reportUnknownMemberType]

    fig.canvas.mpl_connect("button_press_event",lambda event:
                           _on_click(event,ax,tilemap,room_count))

    plt.show()                                                                                  #pyright: ignore[reportUnknownMemberType]
    return

def _time_test(count: int) -> None:
    """
    Timed entry point to program
    """
    print("\033c", end="")
    if count < 1:
        return
    np_rng = np.random.default_rng()
    rand_rng = Random()
    total_time = 0.0
    for _ in range(count):
        start = clock()
        _ = dungeon_map_generator(np_rng, rand_rng)
        time = (clock()-start)*1e-6
        total_time += time
    print(f"Run count: {count}\nTotal Time: {total_time:.6f}\nAverage Time: {total_time/count:.6f}")
    return

def _main() -> None:
    """
    Non-Timed entry point to program
    """
    print("\033c", end="")

    user_input = input("Input Seed: ")
    debug_seed = int(user_input) if user_input else None
    np_rng = np.random.default_rng(debug_seed)
    rand_rng = Random(debug_seed)
  
    
    tilemap = dungeon_map_generator(np_rng, rand_rng)

    room_count = np.count_nonzero(tilemap)
    print(f"Dungeon contains {room_count} rooms")
    _debug(tilemap, room_count)
    return
#endregion

if __name__ == "__main__":
    if arg_parser():
        _time_test(1)
    else:
        _main()