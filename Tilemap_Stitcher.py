"""
**Tilemap Stitcher**

In File Entry Point: _main() or _time_test()
Import Entry Point: dungeon_generator()
"""

from random import Random
from time import perf_counter_ns as clock

import numpy as np
from numpy import uint8
from numpy.typing import NDArray as array

from Debug_Tools import timeit, arg_parser, debug_render
from Dungeon_Generator import dungeon_map_generator
from Room_Generator import room_map_generator, Const

@timeit
def dungeon_generator(
        np_rng: np.random.Generator,
        rand_rng: Random
) -> tuple[array[uint8], array[uint8]]:
    """
    Generates a complete dungeon tilemap by stitching together individual
    room tilemaps for each active room in the dungeon map.

    Parameters
    ----------
    np_rng : np.random.Generator
        NumPy random generator used for room generation.
    rand_rng : Random
        Python random generator used for room generation.

    Returns
    -------
    tilemap : array[uint8]
        2D array of the full dungeon, with each room occupying a
        ROOM_SIZE x ROOM_SIZE region.
    theme_map : array[uint8]
        2D array of the same dimensions as the dungeon map, where each
        value is the Theme of the room at that position, or 0 for
        inactive tiles.

    Notes
    -----
    - dungeon_map_generator() is called first to produce the high-level
      room layout.
    - The output tilemap is sized at dungeon_map.shape * ROOM_SIZE in
      both dimensions.
    - Each active tile in the dungeon map is passed to room_map_generator()
      to produce its room tilemap, which is then written into the
      corresponding region of the full tilemap.
    - Inactive tiles (value 0) are skipped, leaving their regions as
      all zeros.
    - The room shape is discarded; only the theme is stored in theme_map.
    """
    dungeon_map = dungeon_map_generator(np_rng, rand_rng)
    h, w = dungeon_map.shape
    multiplier = Const.ROOM_SIZE
    tilemap = np.zeros((h * multiplier, w * multiplier), dtype = uint8)
    theme_map = np.zeros((h,w), dtype = uint8)

    for (row, col), val in np.ndenumerate(dungeon_map):
        if val == 0:
            continue
        room_tilemap, _, theme = room_map_generator(
            int(val), np_rng, rand_rng
        )
        y = row * multiplier
        x = col * multiplier
        tilemap[y:y + multiplier, x:x + multiplier] = room_tilemap
        theme_map[row,col] = theme
    return tilemap, theme_map

#region DEBUG
def _get_tile_value(value: int) -> tuple[str,str]:
    """
    Formats a raw tile value as a Const enum name for debug display.

    Parameters
    ----------
    value : int
        Raw tile value to format.

    Returns
    -------
    label : str
        Display label for the formatted value: "Tile".
    name : str
        Name of the Const enum member corresponding to value.
    """
    return ("Tile", Const(value).name)

def _debug(tilemap: array[uint8]) -> None:
    """
    Renders an interactive debug visualization of the full dungeon tilemap.

    Parameters
    ----------
    tilemap : array[uint8]
        Full dungeon tilemap as produced by dungeon_generator().

    Returns
    -------
    None

    Notes
    -----
    - Uses the same 11-colour palette as Room_Generator, mapping Const
      values 0-10 to colours.
    - Rendered at 7x7 inches to accommodate the larger tilemap size.
    - Clicking a tile prints its position and Const name.
    - Clicking outside the axes prints the tilemap dimensions.
    """
    colours = [
        "black", "white", "gray", "blue",
        "red", "green", "brown", "yellow",
        "orange", "red", "green"
    ]
    info = {"Size": f"{tilemap.shape}"}
    debug_render(
        tilemap, colours, info,
        figsize=(7, 7),
        tile_formatter = _get_tile_value
    )
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
    - The console is cleared at the start using the '\\033c' escape code.
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
        _ = dungeon_generator(np_rng, rand_rng)
        time = (clock()-start)*1e-6
        total_time += time
    print(f"Run count: {count}\n"
          f"Total Time: {total_time:.6f} ms\n"
          f"Average Time: {total_time/count:.6f} ms")
    return

def _main() -> None:
    """
    Non-timed entry point to the program. Prompts for an optional seed,
    generates a full dungeon, and launches the debug visualizer.

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
    - The theme_map is printed to the console before the debug visualizer
      is launched.
    - This is an internal entry point.
    """
    print("\033c", end="")
    user_input = input("Input Seed: ")
    debug_seed = int(user_input) if user_input else None
    np_rng = np.random.default_rng(debug_seed)
    rand_rng = Random(debug_seed)

    tilemap, theme_map = dungeon_generator(np_rng, rand_rng)

    print(f"\033c{theme_map}")
    _debug(tilemap) 
    return
#endregion

if __name__ == "__main__":
    if arg_parser(): _time_test(int(input("Input Testing Count: ")))
    else: _main()