"""
**Dungeon Map Generator**

In File Entry Point: _main() or _time_test()
Import Entry Point: dungeon_map_generator()
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
from collections import deque

from Generator_Helpers import init_tilemap, adj_map, get_direction_strings
from Debug_Tools import timeit, arg_parser

class Const(IntEnum):
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
def room_fill(tilemap: array[uint8], np_rng: np.random.Generator) -> array[uint8]:
    """
    Places random room regions inside an array\n
    All regions will cross the midpoint
    
    Parameters
    ----------
    tilemap : array[uint8]
        Empty 2D array.
    np_rng : np.random.Generator
        numpy seeded random

    Returns
    -------
    tilemap : array[uint8]
        2D array with placed room regions of random size.
    
    Notes
    -----
    For each room:
        - Vertical span is selected so that each room crosses the midpoint,
          with y_start above and y_end below it.
        - Width is inversely proportional to height, with a maximum span of
          16 tiles, keeping taller rooms narrower.
        - Horizontal placement is chosen randomly, with added padding and
          clamped to the interior boundary to avoid edge overflow.
        - The resulting rectangular region is filled with a temporary tile value.
    """
    for _ in range(Const.BOX_COUNT):
        y_start = np_rng.integers(1, Const.MID)
        y_end = np_rng.integers(Const.MID + 2, Const.DUNGEON_SIZE - 1)

        room_height = y_end-y_start
        room_width = 16 - room_height

        x_start = np_rng.integers(1, Const.MID)
        x_end = min(x_start + room_width + 5, Const.DUNGEON_SIZE - 2)

        tilemap[y_start:y_end, x_start:x_end] = Const.TEMP
    return tilemap

@timeit
def room_eroder(tilemap: array[uint8], np_rng: np.random.Generator) -> array[uint8]:
    """
    Performs edge erosion on existing room regions inside tilemap
    Removes fully isolated tiles

    Parameters
    ----------
    tilemap : array[uint8]
        2D array with room regions placed.
    np_rng : np.random.Generator
        numpy seeded random

    Returns
    -------
    tilemap : array[uint8]
        2D array with room edges eroded for smoother generation.

    Notes
    -----
    Erosion process:
        - An empty neighbor_map is initialized to track adjacency counts for each tile.
        - On each erosion pass, adj_map() scans the tilemap and populates neighbor_map
          with the number of filled neighbors each tile has.
        - Tiles with exactly 2 neighbors (walls) are removed with 50% probability.
        - Tiles with exactly 3 neighbors (corners) are removed with 10% probability.
        - After all passes, a final scan removes any remaining fully isolated tiles
          (tiles with 0 neighbors).
    """
    neighbor_map = np.empty_like(tilemap, dtype=uint8)

    for _ in range(Const.ERODE_COUNT):
        adj_map(tilemap, neighbor_map)

        mask_2 = (neighbor_map == 2)
        tilemap[mask_2 & (np_rng.random(mask_2.shape, dtype = np.float32) < 0.5)] = Const.NO_ROOM

        mask_3 = (neighbor_map == 3)
        tilemap[mask_3 & (np_rng.random(mask_3.shape, dtype = np.float32) < 0.1)] = Const.NO_ROOM

    adj_map(tilemap, neighbor_map)
    tilemap[neighbor_map == 0] = Const.NO_ROOM
    return tilemap

@timeit
def get_possible_connections(tilemap: array[uint8]) -> array[uint8]:
    """
    Converts tilemap into map of adjacent tile count

    Parameters
    ----------
    tilemap : array[uint8]
        2D array containing active and inactive tiles.

    Returns
    -------
    tilemap : array[uint8]
        2D array with borders excluded, containing number
        of active tiles adjacent to each tile.

    Notes
    -----
    - The tilemap is first converted to a boolean array.
    - Each of the four orthogonal neighbors is checked using array slicing, and the
      result is encoded into a specific bit position via bitwise shift.
    - All four directional checks are combined with bitwise OR to form a single
      bitmask per tile.
    - The result is masked by the initial tile values, so inactive tiles always
      produce a connection value of 0.
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
    randoms : array[uint8]
        Array with random values

    Notes
    -----
    Each value is determined by a random float in [0.0, 1.0) with the
    following probability distribution:
        Value 1 : 55% chance  (float < 0.55)
        Value 2 : 25% chance  (0.55 <= float < 0.80)
        Value 3 : 20% chance  (float >= 0.80)

    The array is initialized to all 1s, then values are overwritten
    upward in threshold order so that higher thresholds take priority.
    """
    r = np_rng.random(count, dtype = np.float32)

    randoms = np.ones(count, dtype = uint8)
    randoms[r >= 0.55] = 2
    randoms[r >= 0.80] = 3
    return randoms

@timeit
def room_connector(tilemap: array[uint8], np_rng: np.random.Generator, rand_rng: Random) -> array[uint8]:
    """
    Connects adjacent active rooms across the given tilemap by writing
    directional connection bits into each active tile.

    Parameters
    ----------
    tilemap : array[uint8]
        2D array with active rooms (bit 4).
    np_rng : np.random.Generator
        numpy seeded random, used for weighted connection counts.
    rand_rng : Random
        Python random generator, used for selecting which directions to connect.

    Returns
    -------
    tilemap : array[uint8]
        2D array with rooms connected (bits 0-3).

    Notes
    -----
    Each tile's value uses bits as flags:
        Bit 0 (value 1)  : Connected upward
        Bit 1 (value 2)  : Connected right
        Bit 2 (value 4)  : Connected downward
        Bit 3 (value 8)  : Connected left
        Bit 4 (value 16) : Active room tile

    Connection process:
        - get_possible_connections() builds a map of which orthogonal neighbors
          are active for each tile.
        - room_random() pre-generates a weighted count (1, 2, or 3) for each
          active tile, determining how many connections it will make.
        - Each active tile's available directions are gathered from the
          connection map. If the tile has fewer available directions than its
          target count, all available directions are used.
        - Chosen directions are written as bits into both the current tile and
          its neighbor in the opposite direction, keeping connections two-way.
    """
    connection_map = get_possible_connections(tilemap)
    H, W = tilemap.shape
    active_count = np.count_nonzero(tilemap != 0)
    connection_counts = room_random(np_rng, active_count)
    MASK_TO_INDICES = tuple(tuple(i for i in range(4) if mask & (1 << i)) for mask in range(16))
    DIR_BITS = (1,2,4,8)
    DY_DX = ((-1,0),(0,1),(1,0),(0,-1))
    OPP_BITS = (4,8,1,2)
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

            indices = MASK_TO_INDICES[mask]
            n = len(indices)

            connect_count = connection_counts[connection_index]
            connection_index += 1

            if connect_count >= n:
                chosen = indices
            elif connect_count == 1:
                chosen = (indices[randrange(n)],)
            else:
                chosen = tuple(rand_rng.sample(indices, connect_count))

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
    Reduces the size of the tilemap to the smallest possible bounding box
    without removing any active tiles.

    Parameters
    ----------
    tilemap : array[uint8]
        2D array with rooms in final positions.

    Returns
    -------
    trimmed_tilemap : array[uint8]
        2D array containing only the bounding box of the active room positions.

    Notes
    -----
    - Rows and columns are scanned independently to find which contain at
      least one active tile.
    - numpy.ix_() is used to select only the rows and columns that contain
      active tiles, cropping out any empty border rows or columns.
    - The result is the tightest possible rectangle that still contains all
      active tiles.
    """
    active_rows = np.any(tilemap != 0, axis=1)
    active_cols = np.any(tilemap != 0, axis=0)
    trimmed_tilemap = tilemap[np.ix_(active_rows, active_cols)]
    return trimmed_tilemap

@timeit
def room_clear(tilemap: array[uint8]) -> array[uint8]:
    """
    Removes all disconnected room groups from the tilemap, keeping only
    the largest connected group.

    Parameters
    ----------
    tilemap : array[uint8]
        2D array with room connections encoded in bits 0-3:

            Bit 0 (value 1) : North exit
            Bit 1 (value 2) : East exit
            Bit 2 (value 4) : South exit
            Bit 3 (value 8) : West exit

    Returns
    -------
    tilemap : array[uint8]
        2D array with all tiles not belonging to the largest connected
        group set to 0.

    Notes
    -----
    - All active (non-zero) tile coordinates are collected into a set
      at the start.
    - Connected groups are discovered by iterative flood fill, starting
      from an arbitrary unvisited tile each time.
    - The flood fill only traverses genuine connections by checking
      directional bits, rather than just spatial adjacency.
    - A separate visited set is maintained to prevent tiles from being
      added to the queue more than once.
    - After all groups are found, the largest is kept and all other
      active tiles are zeroed out in a single vectorised numpy operation.
    """
    DIR_OFFSETS = ((0,-1,0),(1,0,1),(2,1,0),(3,0,-1))
    h, w = tilemap.shape
    active_tiles = {(r, c) for r, c in np.argwhere(tilemap != 0).tolist()}
    groups: list[set[tuple[int, int]]] = []
    unvisited = active_tiles.copy()
    while unvisited:
        start = next(iter(unvisited))
        group: set[tuple[int, int]] = set()
        visited: set[tuple[int, int]] = {start}
        queue = deque([start])
        while queue:
            y, x = queue.popleft()
            group.add((y, x))
            val = tilemap[y, x]
            for bit, dy, dx in DIR_OFFSETS:
                if val & (1 << bit):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        queue.append((ny, nx))
        groups.append(group)
        unvisited -= group
    
    if groups:
        largest = max(groups, key=len)
        to_remove = np.array(list(active_tiles - largest), dtype = np.int32).reshape(-1,2)
        tilemap[to_remove[:, 0], to_remove[:, 1]] = 0
    return tilemap

@timeit
def dungeon_map_generator(np_rng: np.random.Generator, rand_rng: Random) -> array[uint8]:
    """
    Generates a complete dungeon tilemap using all intermediary functions.

    Parameters
    ----------
    np_rng : np.random.Generator
        NumPy random generator used for deterministic numeric operations.
    rand_rng : Random
        Python random generator used for non-NumPy randomness.

    Returns
    -------
    tilemap : array[uint8]
        Final dungeon map array.

    Notes
    -----
    Generation pipeline:
        1. init_tilemap()   : Allocates an empty 2D array of size DUNGEON_SIZE.
        2. room_fill()      : Places random overlapping room regions, all
                              crossing the vertical midpoint.
        3. room_eroder()    : Smooths room edges by probabilistically removing
                              tiles with few orthogonal neighbors, then removes
                              any fully isolated tiles.
        4. Bit shift (<<= 4): Moves all active tile values into bit 4, vacating
                              bits 0-3 to store directional connection data.
        5. room_connector() : Connects adjacent active tiles by writing
                              orthogonal connection bits (bits 0-3) into each tile.
        6. room_clear()     : Removes isolated pairs of tiles that have no
                              connections beyond each other.
        7. tilemap_trim()   : Crops the array to the tightest bounding box
                              that still contains all active tiles.
    """
    tilemap = init_tilemap(Const.DUNGEON_SIZE)
    tilemap = room_fill(tilemap, np_rng)
    tilemap = room_eroder(tilemap, np_rng)
    tilemap <<= 4
    tilemap = room_connector(tilemap, np_rng, rand_rng)
    tilemap = room_clear(tilemap)
    tilemap = tilemap_trim(tilemap)
    return tilemap

#region DEBUG
def _make_exit_map(tilemap: array[uint8]) -> array[uint8]:
    """
    Converts a tilemap into a display map showing the number of exits
    each tile has.

    Parameters
    ----------
    tilemap : array[uint8]
        2D array to be converted.

    Returns
    -------
    debug_map : array[uint8]
        2D array of the same dimensions, where each value represents the
        number of active bits (exits) in the corresponding tile.

    Notes
    -----
    - Each tile's uint8 value is unpacked into its 8 individual bits using
      numpy.unpackbits().
    - The bits are summed across the bit axis, producing a count of how many
      bits are set (i.e. how many exits the tile has).
    """
    debug_map = np.unpackbits(tilemap[:, :, np.newaxis], axis=-1).sum(axis=-1).astype(np.uint8)
    return debug_map

def _on_click(event: Event, ax: Axes, tilemap: array[uint8], room_count: int) -> None:
    """
    Local handler for debug click events on the dungeon display.

    Parameters
    ----------
    event : Event
        matplotlib click event.
    ax : Axes
        matplotlib graph axes.
    tilemap : array[uint8]
        tilemap of the dungeon.
    room_count : int
        number of rooms in the dungeon.

    Returns
    -------
    None

    Notes
    -----
    - Only responds to MouseEvent types; all other event types are ignored.
    - If the click lands inside the axes on a valid tile:
        - The clicked pixel coordinates are rounded to the nearest tile index.
        - get_direction_strings() is used to decode the tile's exit bits into
          human-readable direction names.
        - The tile's row/column position, raw value, and exit directions are
          printed to the console.
    - If the click lands outside the axes, the total room count is printed
      instead.
    """
    if not isinstance(event, MouseEvent): return
    if event.inaxes is ax and event.xdata is not None and event.ydata is not None:
        col = int(event.xdata+0.5)
        row = int(event.ydata+0.5)
        if 0 <= row < tilemap.shape[0] and 0 <= col < tilemap.shape[1]:
            dirs = get_direction_strings(tilemap[row,col])
            print(f"\033cTile Clicked: {row}, {col}\n"
                  f"Tile Value: {tilemap[row,col]}\n"
                  f"Exits: {", ".join(dirs)}")
    else: print(f"\033cDungeon contains {room_count} rooms")
    return

def _debug(tilemap: array[uint8], room_count: int) -> None:
    """
    Renders an interactive debug visualization of the dungeon tilemap.

    Parameters
    ----------
    tilemap : array[uint8]
        Dungeon tilemap with room connection bits.
    room_count : int
        Number of active rooms in the dungeon.

    Returns
    -------
    None

    Notes
    -----
    Colour mapping:
        The exit count map produced by _make_exit_map() maps each tile's
        bit count to a colour via a fixed 6-colour palette:
            0 bits set : White  (inactive tile)
            1 bit set  : Black  (impossible under normal generation logic)
            2 bits set : Green  (active tile with 1 connection: bit 4 + 1 exit)
            3 bits set : Blue   (active tile with 2 connections: bit 4 + 2 exits)
            4 bits set : Red    (active tile with 3 connections: bit 4 + 3 exits)
            5 bits set : Yellow (active tile with 4 connections: bit 4 + all 4 exits)

    Display setup:
        - _make_exit_map() converts the raw tilemap into a per-tile exit
          count array for display.
        - The figure is rendered at 5x5 inches at 120 DPI.
        - The matplotlib toolbar is hidden for a cleaner debug window.

    Grid and axes:
        - Minor ticks are placed at half-integer positions to draw grid
          lines between tiles rather than through them.
        - All tick marks and axis labels are hidden, leaving only the
          colour grid visible.

    Interactivity:
        - A click event listener is connected to the figure, delegating
          all click handling to _on_click().
        - Clicking on a tile prints its position, raw value, and exit
          directions.
        - Clicking outside the axes prints the room count.

    Type checking:
        - Several matplotlib calls are marked with pyright: ignore
          [reportUnknownMemberType] due to false positives from incomplete
          matplotlib type stubs. The argument and return types are otherwise
          fully resolved and type-safe.
    """
    debug_map = _make_exit_map(tilemap)
    colours = ["white", "black", "green", "blue", "red", "yellow"]
    cmap = ListedColormap(colours)
    norm = BoundaryNorm(range(len(colours)+1), cmap.N)
    rows, cols = debug_map.shape
    rcParams["toolbar"]="None"

    fig, ax = plt.subplots(figsize = (5,5), dpi = 120)                                          #pyright: ignore[reportUnknownMemberType]
    ax.imshow(debug_map,cmap=cmap,norm=norm,interpolation="nearest")                            #pyright: ignore[reportUnknownMemberType]
    ax.grid(which="minor", color="white", linewidth=0.5)                                        #pyright: ignore[reportUnknownMemberType]
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)  #pyright: ignore[reportUnknownMemberType]
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)                                         #pyright: ignore[reportUnknownMemberType]
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)                                         #pyright: ignore[reportUnknownMemberType]

    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title("DEBUG Window")
    fig.canvas.mpl_connect("button_press_event",lambda event:
                           _on_click(event,ax,tilemap,room_count))

    plt.show()                                                                                  #pyright: ignore[reportUnknownMemberType]
    return

def _time_test(count: int) -> None:
    """
    Runs the dungeon generator a specified number of times and reports
    timing statistics.

    Parameters
    ----------
    count : int
        Number of generation runs to perform. If less than 1, returns
        immediately without running.

    Returns
    -------
    None

    Notes
    -----
    - The console is cleared at the start of each run using the '\\033c'
      escape code.
    - Fresh random generators are created at the start of each run to
      ensure every run is fully independent and representative of a real
      cold-start generation.
    - Each run is timed individually using clock(), with the raw result
      converted from nanoseconds to milliseconds via multiplication by 1e-6.
    - After all runs complete, the following statistics are printed:
        - Total run count
        - Total elapsed time across all runs
        - Average time per run
    - This is an internal entry point.
    """
    print("\033c", end="")
    if count < 1: return
    total_time = 0.0
    for _ in range(count):
        np_rng = np.random.default_rng()
        rand_rng = Random()
        start = clock()
        _ = dungeon_map_generator(np_rng, rand_rng)
        time = (clock()-start)*1e-6
        total_time += time
    print(f"Run count: {count}\nTotal Time: {total_time:.6f} ms\nAverage Time: {total_time/count:.6f} ms")
    return

def _main() -> None:
    """
    Non-timed entry point to the program. Prompts for an optional seed,
    generates a dungeon, and launches the debug visualizer.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    - The console is cleared at the start using the '\\033c' escape code.
    - The user is prompted for an optional integer seed:
        - If a seed is provided, it is used to initialize both random
          generators, making the run fully reproducible.
        - If no seed is entered, both generators are initialized without
          a seed, producing a random result each run.
    - dungeon_map_generator() is called to produce the final tilemap.
    - The number of active rooms is counted as the number of non-zero
      tiles in the tilemap.
    - _debug() is called to launch the interactive visualization window.
    - This is an internal entry point.
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
    if arg_parser(): _time_test(int(input("Input Testing Count: ")))
    else: _main()