"""
**Room Map Generator**
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy import uint8
from numpy.typing import NDArray as array
from matplotlib import rcParams
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event, MouseEvent
from enum import IntEnum

from Constants import Room_Generator_Constants as const
from Generator_Helpers import *

class InvalidRoom(Exception):
    pass

class Tile(IntEnum):
    HOLES      = 0
    WATER      = 1
    TRAPS      = 2
    HEALING    = 3
    CHESTS     = 4
    LOOT_PILES = 5
    MONSTERS   = 6
    BOSS       = 7
    SHRINE     = 8

def get_shape(room_val: int, rng: np.random.Generator) -> tuple[str, set[str]]:
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
    total = sum(weight_list)
    probs = [w / total for w in weight_list]
    shape = rng.choice(shape_list, p=probs)
    return shape, exits

def build_room(tilemap: array[uint8], shape: str, exits: set[str], rng: np.random.Generator) -> array[uint8]:
    half = const.ROOM_SIZE//2
    if "North" in exits:
        tilemap[0:half+1, half-1:half+2] = const.FLOOR
    if "East" in exits:
        tilemap[half-1:half+2, half:const.ROOM_SIZE] = const.FLOOR
    if "South" in exits:
        tilemap[half:const.ROOM_SIZE, half-1:half+2] = const.FLOOR
    if "West" in exits:
        tilemap[half-1:half+2, 0:half+1] = const.FLOOR
    
    match shape:
        case "Dead_End":
            length = rng.integers(1,5)
            tilemap[half-length:half+length+1, half-length:half+length+1] = const.WALL
        case "Boss_Room":
            tilemap[1:-1, 1:-1] = const.FLOOR
        case "Small_Room":
            tilemap[half-3:half+4, half-3:half+4] = const.FLOOR
        case "Connection":
            tilemap[half-1:half+2, half-1:half+2] = const.FLOOR
        case "Large_Room":
            tilemap[half-6:half+7, half-6:half+7] = const.FLOOR
        case "Corner":
            mapping: dict[frozenset[str], tuple[slice, slice]] = {
                frozenset(("North", "West")):  (slice(1, half+2),      slice(1, half+2)),
                frozenset(("North", "East")):  (slice(1, half+2),      slice(half-1, -1)),
                frozenset(("South", "West")):  (slice(half-1, -1),     slice(1, half+2)),
                frozenset(("South", "East")):  (slice(half-1, -1),     slice(half-1, -1)),
            }
            pair = frozenset(exits)
            if pair in mapping: r, c = mapping[pair]; tilemap[r, c] = const.FLOOR
            else: tilemap[half-3:half+4, half-3:half+4] = const.FLOOR
        case "Half":
            if "North" not in exits:
                tilemap[half:-1, 1:-1] = const.FLOOR
            if "East" not in exits:
                tilemap[1:-1, 1:half] = const.FLOOR
            if "South" not in exits:
                tilemap[1:half, 1:-1] = const.FLOOR
            if "West" not in exits:
                tilemap[1:-1, half:-1] = const.FLOOR
        case _:
            raise InvalidRoom(f"Room builder does not support shape {shape}")
    return tilemap

def get_theme(shape: str, rng: np.random.Generator) -> str:
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
    total = sum(weight_list)
    probs = [w / total for w in weight_list]
    theme = rng.choice(theme_list, p=probs)
    return theme

def scan_tilemap(tilemap: array[uint8], require: set[int] | None = None, block: set[int] | None = None,
                 bias: set[int] | None = None, place_on: set[int] = {1}) -> list[tuple[int,int]]:
    available_grid = np.isin(tilemap,list(place_on))

    if require is not None:
        available_grid &= (adj_map(tilemap, target = require, iso = False) != 0)
    if block is not None:
        available_grid &= (adj_map(tilemap, target = block, iso = False) == 0)
    
    available_list = np.argwhere(available_grid)
    if bias is not None:
        bias_grid = available_grid & (adj_map(tilemap, target = bias) != 0)
        bias_list = np.argwhere(bias_grid)
        if bias_list.size > 0:
            available_list = np.concatenate((available_list,bias_list),axis = 0)
    return [tuple(x) for x in available_list]

def populate_tilemap(tilemap: array[uint8], theme: str, rng: np.random.Generator) -> array[uint8]:
    feature_order = (
        Tile.HOLES,
        Tile.WATER,
        Tile.TRAPS,
        Tile.HEALING,
        Tile.CHESTS,
        Tile.LOOT_PILES,
        Tile.MONSTERS,
        Tile.BOSS,
        Tile.SHRINE
    )
    def R(num: int = 0) -> int: return rand(0,1)+num
    def T(num: int = 0) -> int: return rand(0,2)+num
    population_dict: dict[str, dict[Tile, int]] = {
        "DE_Trapped":  {Tile.HOLES: 1, Tile.WATER: R(), Tile.TRAPS: 3},
        "DE_Treasure": {Tile.TRAPS: 1, Tile.CHESTS: 1, Tile.LOOT_PILES: 2, Tile.MONSTERS: 1},
        "DE_Healthy":  {Tile.HEALING: 1},
        "DE_Guarded":  {Tile.MONSTERS: 1},

        "SR_Trapped":  {Tile.HOLES: 1, Tile.TRAPS: T(3), Tile.LOOT_PILES: 1, Tile.MONSTERS: 1},
        "SR_Treasure": {Tile.TRAPS: R(1), Tile.CHESTS: 2, Tile.LOOT_PILES: 3},
        "SR_Guarded":  {Tile.WATER: R(), Tile.TRAPS: 1, Tile.LOOT_PILES: 1, Tile.MONSTERS: 2},
        "SR_Chaos":    {Tile.HOLES: 2, Tile.WATER: R(), Tile.TRAPS: 3, Tile.CHESTS: 1, Tile.LOOT_PILES: 2, Tile.MONSTERS: 3, Tile.SHRINE: 1},
        "SR_Basic":    {Tile.TRAPS: R(), Tile.LOOT_PILES: R()},

        "CN_Trapped":  {Tile.HOLES: 1, Tile.TRAPS: T(1), Tile.LOOT_PILES: 1},
        "CN_Guarded":  {Tile.MONSTERS: 1},
        "CN_Basic":    {Tile.LOOT_PILES: R()},

        "LR_Trapped":  {Tile.HOLES: 2, Tile.WATER: 1, Tile.TRAPS: T(3), Tile.LOOT_PILES: 2, Tile.MONSTERS: 1},
        "LR_Treasure": {Tile.TRAPS: 1, Tile.CHESTS: 2, Tile.LOOT_PILES: 3, Tile.MONSTERS: 1},
        "LR_Healthy":  {Tile.HEALING: 1},
        "LR_Guarded":  {Tile.WATER: R(), Tile.TRAPS: 1, Tile.CHESTS: 1, Tile.LOOT_PILES: 1, Tile.MONSTERS: 3},
        "LR_Chaos":    {Tile.HOLES: 2, Tile.WATER: 1, Tile.TRAPS: 3, Tile.CHESTS: 2, Tile.LOOT_PILES: 3, Tile.MONSTERS: T(2), Tile.SHRINE: 1},
        "LR_Basic":    {Tile.TRAPS: R(1), Tile.LOOT_PILES: R()},

        "CR_Trapped":  {Tile.HOLES: 1, Tile.TRAPS: T(2), Tile.LOOT_PILES: 1},
        "CR_Treasure": {Tile.TRAPS: 1, Tile.CHESTS: 1, Tile.LOOT_PILES: 3, Tile.MONSTERS: 1},
        "CR_Guarded":  {Tile.WATER: R(), Tile.TRAPS: 1, Tile.LOOT_PILES: 1, Tile.MONSTERS: 2},
        "CR_Chaos":    {Tile.HOLES: R(), Tile.WATER: 1, Tile.TRAPS: 3, Tile.CHESTS: T(), Tile.LOOT_PILES: 3, Tile.MONSTERS: R(2), Tile.SHRINE: 1},
        "CR_Basic":    {Tile.TRAPS: R(), Tile.LOOT_PILES: R()},

        "HR_Trapped":  {Tile.HOLES: 1, Tile.TRAPS: T(2), Tile.LOOT_PILES: 1},
        "HR_Treasure": {Tile.TRAPS: 1, Tile.CHESTS: 1, Tile.LOOT_PILES: 3, Tile.MONSTERS: 1},
        "HR_Guarded":  {Tile.WATER: R(), Tile.TRAPS: 1, Tile.LOOT_PILES: 1, Tile.MONSTERS: 2},
        "HR_Chaos":    {Tile.HOLES: R(), Tile.WATER: 1, Tile.TRAPS: 3, Tile.MONSTERS: R(2), Tile.SHRINE: 1, Tile.CHESTS: T(), Tile.LOOT_PILES: 3},
        "HR_Basic":    {Tile.TRAPS: R(), Tile.LOOT_PILES: R()},

        "BR_Hoard":    {Tile.CHESTS: 3, Tile.LOOT_PILES: 9, Tile.BOSS: 1, Tile.SHRINE: 1},
        "BR_Wizard":   {Tile.CHESTS: 4, Tile.LOOT_PILES: 3, Tile.BOSS: 1, Tile.SHRINE: 1},
        "BR_Weak":     {Tile.TRAPS: R(), Tile.CHESTS: 1, Tile.LOOT_PILES: 2, Tile.MONSTERS: 1, Tile.BOSS: 1},
        "BR_Strong":   {Tile.HEALING: R(), Tile.CHESTS: T(1), Tile.LOOT_PILES: 5, Tile.BOSS: 1, Tile.SHRINE: 1},
        "BR_Guarded":  {Tile.TRAPS: 1, Tile.CHESTS: 2, Tile.LOOT_PILES: 3, Tile.MONSTERS: 2, Tile.BOSS: 1, Tile.SHRINE: 1},
        "BR_Double":   {Tile.CHESTS: 3, Tile.LOOT_PILES: 5, Tile.BOSS: 2, Tile.SHRINE: 1},

        "Empty": {}
    }
    pop_vals = population_dict[theme]
    for feature in feature_order:
        count = pop_vals.get(feature)
        if count:
            match feature:
                case Tile.HOLES:
                    available_list = scan_tilemap(tilemap, block = {const.WALL, const.WATER, const.LOOT_PILE})
                    indices = rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = const.HOLE
                case Tile.WATER:
                    available_list = scan_tilemap(tilemap, block = {const.CHEST, const.LOOT_PILE, const.HOLE})
                    indices = rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = const.WATER
                case Tile.TRAPS:
                    available_list = scan_tilemap(tilemap, block = {const.TRAP, const.HEALING_STATION, const.SHRINE})
                    indices = rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = const.TRAP
                case Tile.HEALING:
                    available_list = scan_tilemap(tilemap, require = {const.FLOOR}, place_on = {const.WALL})
                    indices = rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = const.HEALING_STATION
                case Tile.CHESTS:
                    available_list = scan_tilemap(tilemap, bias = {const.LOOT_PILE, const.WALL})
                    indices = rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = const.CHEST
                case Tile.LOOT_PILES:
                    available_list = scan_tilemap(tilemap, bias = {const.CHEST, const.LOOT_PILE}, block = {const.WATER, const.HOLE})
                    for _ in range(count):
                        if len(available_list) > 0:
                            i = rng.integers(0, len(available_list))
                            row, col = available_list[i]
                            tilemap[row,col] = const.LOOT_PILE
                            available_list = np.delete(available_list, i, axis=0).astype(np.int32)
                case Tile.MONSTERS:
                    available_list = scan_tilemap(tilemap, block = {const.BOSS_SPAWNER, const.HEALING_STATION, const.SHRINE})
                    indices = rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = const.MONSTER_SPAWNER
                case Tile.BOSS:
                    available_list = scan_tilemap(tilemap, block = {const.MONSTER_SPAWNER, const.HEALING_STATION, const.SHRINE})
                    indices = rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = const.BOSS_SPAWNER
                case Tile.SHRINE:
                    available_list = scan_tilemap(tilemap, require = {const.FLOOR}, place_on = {const.WALL})
                    indices = rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = const.SHRINE
    return tilemap

def room_map_generator(room_val: int, seed: int | None = None) -> tuple[array[uint8], str, str]:
    if seed is None: rng = np.random.default_rng()
    else: rng = np.random.default_rng(seed)
    tilemap = init_tilemap(const.ROOM_SIZE)
    shape, exits = get_shape(room_val, rng)
    tilemap = build_room(tilemap, shape, exits, rng)
    theme = get_theme(shape, rng)
    tilemap = populate_tilemap(tilemap, theme, rng)
    return tilemap, shape, theme

#region DEBUG
def _on_click(event: Event, ax: Axes, tilemap: array[uint8], time: float, shape: str, theme: str) -> None:
    if not isinstance(event, MouseEvent): return
    if event.inaxes is ax and event.xdata is not None and event.ydata is not None:
        col = int(event.xdata+0.5)
        row = int(event.ydata+0.5)
        if 0 <= row < tilemap.shape[0] and 0 <= col < tilemap.shape[1]:
            print(f"\033cProgram ran in {time} milliseconds\nShape: {shape}\nTheme: {theme}\n"+
                f"Tile Clicked: {row}, {col}\n"+
                f"Tile Value: {const(tilemap[row,col]).name}")
    else:
        print(f"\033cProgram ran in {time} milliseconds\nShape: {shape}\nTheme: {theme}")
    return

def _debug(tilemap: array[uint8], time: float, shape: str, theme: str) -> None:
    debug_map = tilemap

    colours = ["black", "white", "gray", "blue", "red", "green", "brown", "yellow", "orange"]

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
    ax.grid(which="minor", color="black", linewidth=0.5)                                        #type: ignore[reportUnknownMemberType]
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)  #type: ignore[reportUnknownMemberType]
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)                                         #type: ignore[reportUnknownMemberType]
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)                                         #type: ignore[reportUnknownMemberType]

    fig.canvas.mpl_connect("button_press_event",lambda event:
                           _on_click(event,ax,tilemap,time,shape,theme))

    plt.show()                                                                                  #type: ignore[reportUnknownMemberType]
    return

def _main() -> None:
    from time import perf_counter_ns as clock
    print("\033c", end="")

    debug_room_val = int(input("Input Room Value: "))
    debug_seed = int(input("Input Seed: ")) or None
    
    start_time = clock()

    tilemap, shape, theme = room_map_generator(debug_room_val, debug_seed)

    end_time = clock()
    delta_time = (end_time - start_time)/1000000
    print(f"\033cProgram ran in {delta_time} milliseconds\nShape: {shape}\nTheme: {theme}\n")
    _debug(tilemap, delta_time, shape, theme)
    return
#endregion

if __name__ == "__main__":
    _main()