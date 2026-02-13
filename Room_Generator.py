import numpy as np
import matplotlib.pyplot as plt

from numpy import uint8
from numpy.typing import NDArray as array
from random import randint as rand
from random import choice
from random import choices
from matplotlib import rcParams
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.axes import Axes

from Constants import *
from Generator_Helpers import *

class InvalidRoom(Exception):
    pass

def get_shape(room_val: int) -> tuple[str, set[str]]:
    if room_val < 0b10000 or room_val > 0b11111: raise InvalidRoom(f"The get_shape function does not support room_val: {room_val}.")
    exits = get_directions(room_val)
    match len(exits):
        case 1:
            shape_list, weight_list = ["Dead_End", "Boss_Room", "Small_Room"],[35, 15, 50]
        case 2:
            shape_list, weight_list = ["Connection", "Small_Room", "Large_Room", "Corner"], [15, 25, 30, 30]
        case 3:
            shape_list, weight_list = ["Connection", "Small_Room", "Large_Room", "Half"], [20, 20, 30, 30]
        case 4:
            shape_list, weight_list = ["Connection", "Small_Room", "Large_Room"], [20, 30, 50]
        case _:
            raise InvalidRoom(f"The get_shape function does not support rooms with {len(exits)} exits.")
    shape = choices(shape_list, weight_list, k=1)[0]
    return shape, exits

def build_room(tilemap: array[uint8], shape: str, exits: set[str]) -> array[uint8]:
    half = ROOM_SIZE//2
    if "North" in exits:
        tilemap[0:half+1, half-1:half+2] = FLOOR
    if "East" in exits:
        tilemap[half-1:half+2, half:ROOM_SIZE] = FLOOR
    if "South" in exits:
        tilemap[half:ROOM_SIZE, half-1:half+2] = FLOOR
    if "West" in exits:
        tilemap[half-1:half+2, 0:half+1] = FLOOR
    
    match shape:
        case "Dead_End":
            length = rand(1,5)
            tilemap[half-length:half+length+1, half-length:half+length+1] = WALL
        case "Boss_Room":
            tilemap[1:-1, 1:-1] = FLOOR
        case "Small_Room":
            tilemap[half-3:half+4, half-3:half+4] = FLOOR
        case "Connection":
            tilemap[half-1:half+2, half-1:half+2] = FLOOR
        case "Large_Room":
            tilemap[half-6:half+7, half-6:half+7] = FLOOR
        case "Corner":
            mapping = {
                frozenset(("North", "West")):  (slice(1, half+2),      slice(1, half+2)),
                frozenset(("North", "East")):  (slice(1, half+2),      slice(half-1, -1)),
                frozenset(("South", "West")):  (slice(half-1, -1),     slice(1, half+2)),
                frozenset(("South", "East")):  (slice(half-1, -1),     slice(half-1, -1)),
            }
            pair = frozenset(exits)
            if pair in mapping: r, c = mapping[pair]; tilemap[r, c] = FLOOR
            else: tilemap[half-3:half+4, half-3:half+4] = FLOOR
        case "Half":
            if "North" not in exits:
                tilemap[half:-1, 1:-1] = FLOOR
            if "East" not in exits:
                tilemap[1:-1, 1:half] = FLOOR
            if "South" not in exits:
                tilemap[1:half, 1:-1] = FLOOR
            if "West" not in exits:
                tilemap[1:-1, half:-1] = FLOOR
        case _:
            raise Exception
    return tilemap

def get_theme(shape: str) -> str:
    match shape:
        case "Dead_End":
            theme_list, weight_list = ["DE_Trapped","DE_Treasure","DE_Healthy","DE_Guarded","Empty"], [20, 15, 10, 15, 40]
        case "Boss_Room":
            theme_list, weight_list = ["BR_Hoard","BR_Wizard","BR_Weak","BR_Strong","BR_Guarded","BR_Double"], [20,20,20,10,20,10]
        case "Small_Room":
            theme_list, weight_list = ["SR_Trapped","SR_Treasure","SR_Guarded","SR_Chaos","SR_Basic","Empty"], [20,10,15,10,30,15]
        case "Connection":
            theme_list, weight_list = ["CN_Trapped","CN_Guarded","CN_Basic","Empty"], [20,20,30,30]
        case "Large_Room":
            theme_list, weight_list = ["LR_Trapped","LR_Treasure","LR_Healthy","LR_Guarded","LR_Chaos","LR_Basic","Empty"], [20,5,5,15,10,30,15]
        case "Corner":
            theme_list, weight_list = ["CR_Trapped","CR_Treasure","CR_Guarded","CR_Chaos","CR_Basic","Empty"], [20,10,15,10,30,15]
        case "Half":
            theme_list, weight_list = ["HR_Trapped","HR_Treasure","HR_Guarded","HR_Chaos","HR_Basic","Empty"], [20,10,15,10,30,15]
        case _:
            raise InvalidRoom(f"The get_theme function does not support rooms with {shape} shape")
    theme = choices(theme_list, weight_list, k=1)[0]
    return theme

def populate_tilemap(tilemap: array[uint8], theme: str) -> array[uint8]:
    #List Format is [Holes,Water,Traps,Healing,Chests,Loot Piles,Monsters]
    population_dict = {
        "DE_Trapped":   [],
        "DE_Treasure":  [],
        "DE_Healthy":   [],
        "DE_Guarded":   [],
        "SR_Trapped":   [],
        "SR_Treasure":  [],
        "SR_Guarded":   [],
        "SR_Chaos":     [],
        "SR_Basic":     [],
        "CN_Trapped":   [],
        "CN_Guarded":   [],
        "CN_Basic":     [],
        "LR_Trapped":   [],
        "LR_Treasure":  [],
        "LR_Healthy":   [],
        "LR_Guarded":   [],
        "LR_Chaos":     [],
        "LR_Basic":     [],
        "CR_Trapped":   [],
        "CR_Treasure":  [],
        "CR_Guarded":   [],
        "CR_Chaos":     [],
        "CR_Basic":     [],
        "HR_Trapped":   [],
        "HR_Treasure":  [],
        "HR_Guarded":   [],
        "HR_Chaos":     [],
        "HR_Basic":     [],
        "BR_Hoard":     [],
        "BR_Wizard":    [],
        "BR_Weak":      [],
        "BR_Strong":    [],
        "BR_Guarded":   [],
        "BR_Double":    [],
        "Empty":        []
    }
    return tilemap

def room_map_generator(room_val: int) -> tuple[array[uint8], str, str]:
    tilemap = init_tilemap(ROOM_SIZE)
    shape, exits = get_shape(room_val)
    tilemap = build_room(tilemap, shape, exits)
    theme = get_theme(shape)
    return tilemap, shape, theme

def _on_click(event, ax: Axes, tilemap: array[uint8], time: float, shape: str, theme: str) -> None:
    if event.inaxes == ax:
        col = int(event.xdata+0.5)
        row = int(event.ydata+0.5)
        if 0 <= row < tilemap.shape[0] and 0 <= col < tilemap.shape[1]:
            print(f"\033cProgram ran in {time} milliseconds\nShape: {shape}\nTheme: {theme}\n"+
                f"Tile Clicked: {row}, {col}\n"+
                f"Tile Value: {tilemap[row,col]}")
    else:
        print(f"\033cProgram ran in {time} milliseconds\nShape: {shape}\nTheme: {theme}")
    return

def _debug(tilemap: array[uint8], time: float, shape: str, theme: str) -> None:
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

    fig.canvas.mpl_connect("button_press_event",lambda event: _on_click(event, ax, tilemap, time, shape, theme))

    plt.show()
    return

def _main() -> None:
    from time import perf_counter_ns as clock
    print("\033c", end="")

    debug_room_val = int(input("Input Room Value: "))
    
    start_time = clock()

    tilemap, shape, theme = room_map_generator(debug_room_val)

    end_time = clock()
    delta_time = (end_time - start_time)/1000000
    print(f"\033cProgram ran in {delta_time} milliseconds\nShape: {shape}\nTheme: {theme}\n")
    _debug(tilemap, delta_time, shape, theme)
    return

if __name__ == "__main__":
    _main()