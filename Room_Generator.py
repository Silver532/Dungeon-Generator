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
from Generator_Helpers import *

def room_map_generator() -> array[uint8]:
    tilemap = init_tilemap(ROOM_SIZE)
    return tilemap

def _on_click(event, ax: Axes, tilemap: array[uint8], time: float) -> None:
    if event.inaxes == ax:
        col = int(event.xdata+0.5)
        row = int(event.ydata+0.5)
        if 0 <= row < tilemap.shape[0] and 0 <= col < tilemap.shape[1]:
            print(f"\033cProgram ran in {time} milliseconds\n"+
                f"Tile Clicked: {row}, {col}\n"+
                f"Tile Value: {tilemap[row,col]}\n")
    else:
        print(f"\033cProgram ran in {time} milliseconds\n")
    return

def _debug(tilemap: array[uint8], time: float) -> None:
    debug_map = tilemap

    colours = ["black", "white", "gray", "blue", "red", "green", "brown", "yellow"]

    cmap = ListedColormap(colours)
    norm = BoundaryNorm(range(len(colours)+1), cmap.N)

    rcParams["toolbar"]="None"
    fig, ax = plt.subplots(figsize = (5,5), dpi = 120)

    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title("DEBUG Window")
    
    rows, cols = debug_map.shape

    ax.imshow(debug_map,cmap=cmap,norm=norm,interpolation="nearest")
    ax.grid(which="minor", color="black", linewidth=0.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)

    fig.canvas.mpl_connect("button_press_event",lambda event: _on_click(event, ax, tilemap, time))

    plt.show()
    return

def _main() -> None:
    from time import perf_counter_ns as clock
    print("\033c", end="")
    start_time = clock()

    tilemap = room_map_generator()

    end_time = clock()
    delta_time = (end_time - start_time)/1000000
    print(f"Program ran in {delta_time} milliseconds")
    _debug(tilemap, delta_time)
    return

if __name__ == "__main__":
    _main()