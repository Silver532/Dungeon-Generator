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
from time import perf_counter_ns as clock
from enum import IntEnum

from Generator_Helpers import init_tilemap, adj_map
from Debug_Tools import timeit, arg_parser

class InvalidRoom(Exception):
    pass

class const(IntEnum):
    """
    Constants for Room_Generator file
    """
    ROOM_SIZE = 17
    HALF = ROOM_SIZE//2
    WALL = 0
    FLOOR = 1
    HOLE = 2
    WATER = 3
    TRAP = 4
    HEALING_STATION = 5
    CHEST = 6
    LOOT_PILE = 7
    MONSTER_SPAWNER = 8
    BOSS_SPAWNER = 9
    SHRINE = 10

class shape(IntEnum):
    DEAD_END = 1
    BOSS_ROOM = 2
    SMALL_ROOM = 3
    LARGE_ROOM = 4
    CONNECTION = 5
    CORNER = 6
    HALF = 7

_SHAPE_TABLES: dict[int, tuple[list[shape], list[float]]] = {
    1: ([shape.DEAD_END, shape.BOSS_ROOM, shape.SMALL_ROOM],                 [0.35, 0.15, 0.50]),
    2: ([shape.CONNECTION, shape.SMALL_ROOM, shape.LARGE_ROOM, shape.CORNER],[0.15, 0.25, 0.30, 0.30]),
    3: ([shape.CONNECTION, shape.SMALL_ROOM, shape.LARGE_ROOM, shape.HALF],  [0.20, 0.20, 0.30, 0.30]),
    4: ([shape.CONNECTION, shape.SMALL_ROOM, shape.LARGE_ROOM],              [0.20, 0.30, 0.50]),
}

@timeit
def get_shape(room_val: int, rng: np.random.Generator) -> shape:
    """
    Randomly decides room shape dependent on room value

    Parameters
    ----------
    room_val : int
        dungeon room value
    rng : np.random.Generator
        seeded random

    Returns
    -------
    room_shape : shape
        shape of room
    
    """
    if room_val < 0b10000 or room_val > 0b11111: raise InvalidRoom(f"The get_shape function does not support room_val: {room_val}.")
    n = (room_val & 0b01111).bit_count()
    if n not in _SHAPE_TABLES: raise InvalidRoom(f"The get_shape function does not support rooms with {n} exits.")
    shape_list, probs = _SHAPE_TABLES[n]
    room_shape: shape = rng.choice(shape_list, p=probs)
    return room_shape

@timeit
def build_room(tilemap: array[uint8], room_val: int, room_shape: shape, rng: np.random.Generator) -> array[uint8]:
    """
    Builds room exits and shape onto tilemap

    Parameters
    ----------
    tilemap : NDArray[uint8]
        tilemap to build onto
    room_val : int
        dungeon room value
    room_shape : shape
        shape of room
    rng : np.random.Generator
        seeded random

    Returns
    -------
    tilemap : NDArray[uint8]
        tilemap with room outline built
    """
    half = const.HALF
    if 0b00001 & room_val:
        tilemap[0:half+1, half-1:half+2] = const.FLOOR
    if 0b00010 & room_val:
        tilemap[half-1:half+2, half:const.ROOM_SIZE] = const.FLOOR
    if 0b00100 & room_val:
        tilemap[half:const.ROOM_SIZE, half-1:half+2] = const.FLOOR
    if 0b01000 & room_val:
        tilemap[half-1:half+2, 0:half+1] = const.FLOOR
    
    match room_shape:
        case shape.DEAD_END:
            length = rng.integers(2,5, endpoint = True)
            tilemap[half-length:half+length+1, half-length:half+length+1] = const.WALL
        case shape.BOSS_ROOM:
            tilemap[1:-1, 1:-1] = const.FLOOR
        case shape.SMALL_ROOM:
            tilemap[half-3:half+4, half-3:half+4] = const.FLOOR
        case shape.CONNECTION:
            tilemap[half-1:half+2, half-1:half+2] = const.FLOOR
        case shape.LARGE_ROOM:
            tilemap[half-6:half+7, half-6:half+7] = const.FLOOR
        case shape.CORNER:
            match room_val & 0b01111:
                case 0b01001:
                    tilemap[1:half+2, 1:half+2] = const.FLOOR
                case 0b00011:
                    tilemap[1:half+2, half-1:-1] = const.FLOOR
                case 0b01100:
                    tilemap[half-1:-1, 1:half+2] = const.FLOOR
                case 0b00110:
                    tilemap[half-1:-1, half-1:-1] = const.FLOOR
                case _:
                    tilemap[half-3:half+4, half-3:half+4] = const.FLOOR
        case shape.HALF:
            match room_val & 0b01111:
                case 0b01110:
                    tilemap[half:-1, 1:-1] = const.FLOOR
                case 0b01101:
                    tilemap[1:-1, 1:half] = const.FLOOR
                case 0b01011:
                    tilemap[1:half, 1:-1] = const.FLOOR
                case 0b00111:
                    tilemap[1:-1, half:-1] = const.FLOOR
                case _:
                    pass
    return tilemap

@timeit
def get_theme(shape: str, rng: np.random.Generator) -> str:
    """
    Randomly decides room theme dependent on room shape

    Parameters
    ----------
    shape : str
        shape of room
    rng : np.random.Generator

    Returns
    -------
    theme : str
        theme of room
    """
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

@timeit
def scan_tilemap(tilemap: array[uint8], require: set[int] | None = None, block: set[int] | None = None,
                 bias: set[int] | None = None, place_on: set[int] | None = None) -> array[np.int32]:
    """
    Universal tilemap scanner for Room_Generator

    Parameters
    ----------
    tilemap : NDArray[uint8]
        tilemap to scan
    require : set[int] | None = None
        if provided, active checking tiles are limited to values in the set
    block : set[int] | None = None
        if provided, values in this set are blocked from being active checking tiles
    bias : set[int] | None = None
        if provided, values in this set are counted 4 extra times in the available_list
    place_on : set[int] | None = None
        if provided, active placing tiles are limited to values in the set

    Returns
    -------
    available_list : NDArray[np.int32]
        numpy list of valid indeces to place on
    """
    if place_on is None: place_on = {1}
    available_grid = np.isin(tilemap,list(place_on))

    if require is not None:
        available_grid &= (adj_map(tilemap, target = require, iso = False) != 0)
    if block is not None:
        available_grid &= (adj_map(tilemap, target = block, iso = False) == 0)
    
    available_list = np.argwhere(available_grid)
    if bias is not None:
        bias_grid = available_grid & (adj_map(tilemap, target = bias, iso = False) != 0)
        biases = np.argwhere(bias_grid)
        if biases.size > 0:
            bias_list = np.repeat(biases, 4, axis=0)
            available_list = np.concatenate((available_list,bias_list),axis = 0)
    return available_list

@timeit
def populate_tilemap(tilemap: array[uint8], theme: str, rng: np.random.Generator) -> array[uint8]:
    """
    Populates tilemap with features

    Parameters
    ----------
    tilemap : NDArray[uint8]
        tilemap to populate
    theme : str
        theme of room
    rng : np.random.Generator
        seeded random

    Returns
    -------
    tilemap : NDArray[uint8]
        populated tilemap
    """
    feature_order = (
        const.WATER,
        const.HOLE,
        const.HEALING_STATION,
        const.SHRINE,
        const.CHEST,
        const.LOOT_PILE,
        const.TRAP,
        const.BOSS_SPAWNER,
        const.MONSTER_SPAWNER
    )
    def R(num: int = 0) -> int: return rng.integers(0,1)+num
    def T(num: int = 0) -> int: return rng.integers(0,2)+num
    population_dict: dict[str, dict[const, int]] = {
        "DE_Trapped":  {const.HOLE: 1, const.WATER: R(), const.TRAP: 3},
        "DE_Treasure": {const.TRAP: 1, const.CHEST: 1, const.LOOT_PILE: 2, const.MONSTER_SPAWNER: 1},
        "DE_Healthy":  {const.HEALING_STATION: 1},
        "DE_Guarded":  {const.MONSTER_SPAWNER: 1},

        "SR_Trapped":  {const.HOLE: 1, const.TRAP: T(3), const.LOOT_PILE: 1, const.MONSTER_SPAWNER: 1},
        "SR_Treasure": {const.TRAP: R(1), const.CHEST: 2, const.LOOT_PILE: 3},
        "SR_Guarded":  {const.WATER: R(), const.TRAP: 1, const.LOOT_PILE: 1, const.MONSTER_SPAWNER: 2},
        "SR_Chaos":    {const.HOLE: 2, const.WATER: R(), const.TRAP: 3, const.CHEST: 1, const.LOOT_PILE: 2, const.MONSTER_SPAWNER: 3, const.SHRINE: 1},
        "SR_Basic":    {const.TRAP: R(), const.LOOT_PILE: R()},

        "CN_Trapped":  {const.HOLE: 1, const.TRAP: T(1), const.LOOT_PILE: 1},
        "CN_Guarded":  {const.MONSTER_SPAWNER: 1},
        "CN_Basic":    {const.LOOT_PILE: R()},

        "LR_Trapped":  {const.HOLE: 2, const.WATER: 1, const.TRAP: T(3), const.LOOT_PILE: 2, const.MONSTER_SPAWNER: 1},
        "LR_Treasure": {const.TRAP: 1, const.CHEST: 2, const.LOOT_PILE: 3, const.MONSTER_SPAWNER: 1},
        "LR_Healthy":  {const.HEALING_STATION: 1},
        "LR_Guarded":  {const.WATER: R(), const.TRAP: 1, const.CHEST: 1, const.LOOT_PILE: 1, const.MONSTER_SPAWNER: 3},
        "LR_Chaos":    {const.HOLE: 2, const.WATER: 1, const.TRAP: 3, const.CHEST: 2, const.LOOT_PILE: 3, const.MONSTER_SPAWNER: T(2), const.SHRINE: 1},
        "LR_Basic":    {const.TRAP: R(1), const.LOOT_PILE: R()},

        "CR_Trapped":  {const.HOLE: 1, const.TRAP: T(2), const.LOOT_PILE: 1},
        "CR_Treasure": {const.TRAP: 1, const.CHEST: 1, const.LOOT_PILE: 3, const.MONSTER_SPAWNER: 1},
        "CR_Guarded":  {const.WATER: R(), const.TRAP: 1, const.LOOT_PILE: 1, const.MONSTER_SPAWNER: 2},
        "CR_Chaos":    {const.HOLE: R(), const.WATER: 1, const.TRAP: 3, const.CHEST: T(), const.LOOT_PILE: 3, const.MONSTER_SPAWNER: R(2), const.SHRINE: 1},
        "CR_Basic":    {const.TRAP: R(), const.LOOT_PILE: R()},

        "HR_Trapped":  {const.HOLE: 1, const.TRAP: T(2), const.LOOT_PILE: 1},
        "HR_Treasure": {const.TRAP: 1, const.CHEST: 1, const.LOOT_PILE: 3, const.MONSTER_SPAWNER: 1},
        "HR_Guarded":  {const.WATER: R(), const.TRAP: 1, const.LOOT_PILE: 1, const.MONSTER_SPAWNER: 2},
        "HR_Chaos":    {const.HOLE: R(), const.WATER: 1, const.TRAP: 3, const.MONSTER_SPAWNER: R(2), const.SHRINE: 1, const.CHEST: T(), const.LOOT_PILE: 3},
        "HR_Basic":    {const.TRAP: R(), const.LOOT_PILE: R()},

        "BR_Hoard":    {const.CHEST: 3, const.LOOT_PILE: 9, const.BOSS_SPAWNER: 1, const.SHRINE: 1},
        "BR_Wizard":   {const.CHEST: 4, const.LOOT_PILE: 3, const.BOSS_SPAWNER: 1, const.SHRINE: 1},
        "BR_Weak":     {const.TRAP: R(), const.CHEST: 1, const.LOOT_PILE: 2, const.MONSTER_SPAWNER: 1, const.BOSS_SPAWNER: 1},
        "BR_Strong":   {const.HEALING_STATION: R(), const.CHEST: T(1), const.LOOT_PILE: 5, const.BOSS_SPAWNER: 1, const.SHRINE: 1},
        "BR_Guarded":  {const.TRAP: 1, const.CHEST: 2, const.LOOT_PILE: 3, const.MONSTER_SPAWNER: 2, const.BOSS_SPAWNER: 1, const.SHRINE: 1},
        "BR_Double":   {const.CHEST: 3, const.LOOT_PILE: 5, const.BOSS_SPAWNER: 2, const.SHRINE: 1},

        "Empty": {}
    }
    pop_vals = population_dict[theme]
    for feature in feature_order:
        count = pop_vals.get(feature)
        if count:
            match feature:
                case const.HOLE:
                    available_list = scan_tilemap(tilemap, block = {const.WALL, const.WATER, const.LOOT_PILE})
                    indices = rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = const.HOLE
                case const.WATER:
                    available_list = scan_tilemap(tilemap, block = {const.CHEST, const.LOOT_PILE, const.HOLE})
                    indices = rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = const.WATER
                case const.TRAP:
                    available_list = scan_tilemap(tilemap, block = {const.TRAP, const.HEALING_STATION, const.SHRINE})
                    indices = rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = const.TRAP
                case const.HEALING_STATION:
                    available_list = scan_tilemap(tilemap, require = {const.FLOOR}, place_on = {const.WALL})
                    indices = rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = const.HEALING_STATION
                case const.CHEST:
                    available_list = scan_tilemap(tilemap, bias = {const.LOOT_PILE, const.WALL})
                    indices = rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = const.CHEST
                case const.LOOT_PILE:
                    available_list = scan_tilemap(tilemap, bias = {const.CHEST, const.LOOT_PILE}, block = {const.WATER, const.HOLE})
                    for _ in range(count):
                        if len(available_list) > 0:
                            i = rng.integers(0, len(available_list))
                            row, col = available_list[i]
                            tilemap[row,col] = const.LOOT_PILE
                            available_list = np.delete(available_list, i, axis=0).astype(np.int32)
                case const.MONSTER_SPAWNER:
                    available_list = scan_tilemap(tilemap, block = {const.BOSS_SPAWNER, const.HEALING_STATION, const.SHRINE})
                    indices = rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = const.MONSTER_SPAWNER
                case const.BOSS_SPAWNER:
                    available_list = scan_tilemap(tilemap, block = {const.MONSTER_SPAWNER, const.HEALING_STATION, const.SHRINE})
                    indices = rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = const.BOSS_SPAWNER
                case const.SHRINE:
                    available_list = scan_tilemap(tilemap, require = {const.FLOOR}, place_on = {const.WALL})
                    indices = rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = const.SHRINE
    return tilemap

@timeit
def room_map_generator(room_val: int, rng: np.random.Generator) -> tuple[array[uint8], str, str]:
    """
    Handler function to create Room map

    Parameters
    ----------
    room_val : int
        value of room tile to generate
    rng : np.random.Generator
        numpy seeded rng

    Returns
    -------
    tilemap : NDArray[uint8]
        final room tilemap
    shape : str
        shape of room
    theme : str
        theme of room
    """
    tilemap = init_tilemap(const.ROOM_SIZE)
    shape = get_shape(room_val, rng)
    tilemap = build_room(tilemap, shape, exits, rng)
    theme = get_theme(shape, rng)
    tilemap = populate_tilemap(tilemap, theme, rng)
    return tilemap, shape, theme

#region DEBUG
def _on_click(event: Event, ax: Axes, tilemap: array[uint8], shape: str, theme: str) -> None:
    """
    Local handler for debug click events

    Parameters
    ----------
    event : Event
        matplotlib click event
    ax : Axes
        matplotlib graph axes
    tilemap : NDArray[uint8]
        tilemap of the room
    time : float
        number of ms it took for room to generate
    shape : str
        shape of room
    theme : str
        theme of room
    """
    if not isinstance(event, MouseEvent): return
    if event.inaxes is ax and event.xdata is not None and event.ydata is not None:
        col = int(event.xdata+0.5)
        row = int(event.ydata+0.5)
        if 0 <= row < tilemap.shape[0] and 0 <= col < tilemap.shape[1]:
            print(f"\033cShape: {shape}\nTheme: {theme}\n"+
                f"Tile Clicked: {row}, {col}\n"+
                f"Tile Value: {const(tilemap[row,col]).name}")
    else:
        print(f"\033cShape: {shape}\nTheme: {theme}")
    return

def _debug(tilemap: array[uint8], shape: str, theme: str) -> None:
    """
    Local handler for visualization and debugging

    Parameters
    ----------
    tilemap : NDArray[uint8]
        room tilemap
    time : float
        number of ms it took for dungeon to generate
    shape : str
        shape of room
    theme : str
        theme of room
    """
    debug_map = tilemap

    colours = ["black", "white", "gray", "blue", "red", "green", "brown", "yellow", "orange"]

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
    ax.grid(which="minor", color="black", linewidth=0.5)                                        #pyright: ignore[reportUnknownMemberType]
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)  #pyright: ignore[reportUnknownMemberType]
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)                                         #pyright: ignore[reportUnknownMemberType]
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)                                         #pyright: ignore[reportUnknownMemberType]

    fig.canvas.mpl_connect("button_press_event",lambda event:
                           _on_click(event,ax,tilemap,shape,theme))

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
        room_val = np_rng.integers(17, 31, endpoint = True)
        start = clock()
        _ = room_map_generator(room_val, np_rng)
        time = (clock()-start)*1e-6
        total_time += time
    print(f"Run count: {count}\nTotal Time: {total_time:.6f} ms\nAverage Time: {total_time/count:.6f} ms")
    return

def _main() -> None:
    """
    Debug entry point to program
    """
    print("\033c", end="")

    debug_room_val = int(input("Input Room Value: "))
    user_input = input("Input Seed: ")
    debug_seed = int(user_input) if user_input else None
    np_rng = np.random.default_rng(debug_seed)

    tilemap, shape, theme = room_map_generator(debug_room_val, np_rng)

    print(f"\033cShape: {shape}\nTheme: {theme}\n")
    _debug(tilemap, shape, theme)
    return
#endregion

if __name__ == "__main__":
    if arg_parser(): _time_test(int(input("Input Testing Count: ")))
    else: _main()