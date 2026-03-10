"""
**Tilemap Stitcher**

In File Entry Point: _main() or _time_test()
Import Entry Point: dungeon_generator()
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy import uint8
from numpy.typing import NDArray as array
from matplotlib import rcParams
from matplotlib.colors import ListedColormap, BoundaryNorm

from time import perf_counter_ns as clock
from random import Random

from Debug_Tools import timeit, arg_parser
from Room_Generator import room_map_generator, Const
from Dungeon_Generator import dungeon_map_generator

@timeit
def dungeon_generator(np_rng: np.random.Generator, rand_rng: Random) -> tuple[array[uint8], array[uint8]]:
    dungeon_map = dungeon_map_generator(np_rng, rand_rng)
    h, w = dungeon_map.shape
    multiplier = Const.ROOM_SIZE
    tilemap = np.zeros((h * multiplier, w * multiplier), dtype = uint8)
    theme_map = np.zeros((h,w), dtype = uint8)

    for (row, col), val in np.ndenumerate(dungeon_map):
        if val == 0:
            continue
        room_tilemap, _, theme = room_map_generator(int(val), np_rng, rand_rng)
        y = row * multiplier
        x = col * multiplier
        tilemap[y:y + multiplier, x:x + multiplier] = room_tilemap
        theme_map[row,col] = theme
    return tilemap, theme_map

#region DEBUG
def _debug(tilemap: array[uint8]) -> None:
    colours = ["black", "white", "gray", "blue", "red", "green", "brown", "yellow", "orange", "red", "green"]
    cmap = ListedColormap(colours)
    norm = BoundaryNorm(range(len(colours)+1), cmap.N)
    rows, cols = tilemap.shape
    rcParams["toolbar"]="None"
    
    fig, ax = plt.subplots(figsize = (7,7), dpi = 120)                                          #pyright: ignore[reportUnknownMemberType]
    ax.imshow(tilemap,cmap=cmap,norm=norm,interpolation="nearest")                              #pyright: ignore[reportUnknownMemberType]
    ax.grid(which="minor", color="black", linewidth=0.5)                                        #pyright: ignore[reportUnknownMemberType]
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)  #pyright: ignore[reportUnknownMemberType]
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)                                         #pyright: ignore[reportUnknownMemberType]
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)                                         #pyright: ignore[reportUnknownMemberType]

    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title("DEBUG Window")

    plt.show()                                                                                  #pyright: ignore[reportUnknownMemberType]
    return

def _time_test(count: int) -> None:
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
    print(f"Run count: {count}\nTotal Time: {total_time:.6f} ms\nAverage Time: {total_time/count:.6f} ms")
    return

def _main() -> None:
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