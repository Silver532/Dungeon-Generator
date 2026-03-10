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
from collections.abc import Collection
from time import perf_counter_ns as clock
from enum import IntEnum
from random import Random

from Generator_Helpers import init_tilemap, adj_map
from Debug_Tools import timeit, arg_parser

class InvalidRoom(Exception):
    pass

class Const(IntEnum):
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

class Shape(IntEnum):
    DEAD_END = 1
    BOSS_ROOM = 2
    SMALL_ROOM = 3
    LARGE_ROOM = 4
    CONNECTION = 5
    CORNER = 6
    HALF = 7

class Theme(IntEnum):
    EMPTY = 0
    DE_TRAPPED = 1
    DE_TREASURE = 2
    DE_HEALTHY = 3
    DE_GUARDED = 4
    BR_HOARD = 5
    BR_WIZARD = 6
    BR_WEAK = 7
    BR_STRONG = 8
    BR_GUARDED = 9
    BR_DOUBLE = 10
    SR_TRAPPED = 11
    SR_TREASURE = 12
    SR_GUARDED = 13
    SR_CHAOS = 14
    SR_BASIC = 15
    CN_TRAPPED = 16
    CN_GUARDED = 17
    CN_BASIC = 18
    LR_TRAPPED = 19
    LR_TREASURE = 20
    LR_HEALTHY = 21
    LR_GUARDED = 22
    LR_CHAOS = 23
    LR_BASIC = 24
    CR_TRAPPED = 25
    CR_TREASURE = 26
    CR_GUARDED = 27
    CR_CHAOS = 28
    CR_BASIC = 29
    HR_TRAPPED = 30
    HR_TREASURE = 31
    HR_GUARDED = 32
    HR_CHAOS = 33
    HR_BASIC = 34

_SHAPE_TABLES: dict[int, tuple[list[Shape], list[float]]] = {
    1: ([Shape.DEAD_END, Shape.BOSS_ROOM, Shape.SMALL_ROOM],                 [0.35, 0.15, 0.50]),
    2: ([Shape.CONNECTION, Shape.SMALL_ROOM, Shape.LARGE_ROOM, Shape.CORNER],[0.15, 0.25, 0.30, 0.30]),
    3: ([Shape.CONNECTION, Shape.SMALL_ROOM, Shape.LARGE_ROOM, Shape.HALF],  [0.20, 0.20, 0.30, 0.30]),
    4: ([Shape.CONNECTION, Shape.SMALL_ROOM, Shape.LARGE_ROOM],              [0.20, 0.30, 0.50]),
}

_THEME_TABLES: dict[Shape, tuple[list[Theme], list[float]]] = {
    Shape.DEAD_END:   ([Theme.DE_TRAPPED, Theme.DE_TREASURE, Theme.DE_HEALTHY, Theme.DE_GUARDED, Theme.EMPTY], [0.20, 0.15, 0.10, 0.15, 0.40]),
    Shape.BOSS_ROOM:  ([Theme.BR_HOARD, Theme.BR_WIZARD, Theme.BR_WEAK, Theme.BR_STRONG, Theme.BR_GUARDED, Theme.BR_DOUBLE], [0.20, 0.20, 0.20, 0.10, 0.20, 0.10]),
    Shape.SMALL_ROOM: ([Theme.SR_TRAPPED, Theme.SR_TREASURE, Theme.SR_GUARDED, Theme.SR_CHAOS, Theme.SR_BASIC, Theme.EMPTY], [0.20, 0.10, 0.15, 0.10, 0.30, 0.15]),
    Shape.CONNECTION: ([Theme.CN_TRAPPED, Theme.CN_GUARDED, Theme.CN_BASIC, Theme.EMPTY], [0.20, 0.20, 0.30, 0.30]),
    Shape.LARGE_ROOM: ([Theme.LR_TRAPPED, Theme.LR_TREASURE, Theme.LR_HEALTHY, Theme.LR_GUARDED, Theme.LR_CHAOS, Theme.LR_BASIC, Theme.EMPTY], [0.20, 0.05, 0.05, 0.15, 0.10, 0.30, 0.15]),
    Shape.CORNER:     ([Theme.CR_TRAPPED, Theme.CR_TREASURE, Theme.CR_GUARDED, Theme.CR_CHAOS, Theme.CR_BASIC, Theme.EMPTY], [0.20, 0.10, 0.15, 0.10, 0.30, 0.15]),
    Shape.HALF:       ([Theme.HR_TRAPPED, Theme.HR_TREASURE, Theme.HR_GUARDED, Theme.HR_CHAOS, Theme.HR_BASIC, Theme.EMPTY], [0.20, 0.10, 0.15, 0.10, 0.30, 0.15]),
}

@timeit
def get_shape(room_val: int, rand_rng: Random) -> Shape:
    """
    Randomly decides room shape dependent on room value

    Parameters
    ----------
    room_val : int
        dungeon room value
    np_rng : np.random.Generator
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
    room_shape: Shape = rand_rng.choices(shape_list, probs)[0]
    return room_shape

@timeit
def build_room(tilemap: array[uint8], room_val: int, room_shape: Shape, np_rng: np.random.Generator) -> array[uint8]:
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
    np_rng : np.random.Generator
        seeded random

    Returns
    -------
    tilemap : NDArray[uint8]
        tilemap with room outline built
    """
    half = Const.HALF
    if 0b00001 & room_val:
        tilemap[0:half+1, half-1:half+2] = Const.FLOOR
    if 0b00010 & room_val:
        tilemap[half-1:half+2, half:Const.ROOM_SIZE] = Const.FLOOR
    if 0b00100 & room_val:
        tilemap[half:Const.ROOM_SIZE, half-1:half+2] = Const.FLOOR
    if 0b01000 & room_val:
        tilemap[half-1:half+2, 0:half+1] = Const.FLOOR
    
    match room_shape:
        case Shape.DEAD_END:
            length = np_rng.integers(2,5, endpoint = True)
            tilemap[half-length:half+length+1, half-length:half+length+1] = Const.WALL
        case Shape.BOSS_ROOM:
            tilemap[1:-1, 1:-1] = Const.FLOOR
        case Shape.SMALL_ROOM:
            tilemap[half-3:half+4, half-3:half+4] = Const.FLOOR
        case Shape.CONNECTION:
            tilemap[half-1:half+2, half-1:half+2] = Const.FLOOR
        case Shape.LARGE_ROOM:
            tilemap[half-6:half+7, half-6:half+7] = Const.FLOOR
        case Shape.CORNER:
            match room_val & 0b01111:
                case 0b01001:
                    tilemap[1:half+2, 1:half+2] = Const.FLOOR
                case 0b00011:
                    tilemap[1:half+2, half-1:-1] = Const.FLOOR
                case 0b01100:
                    tilemap[half-1:-1, 1:half+2] = Const.FLOOR
                case 0b00110:
                    tilemap[half-1:-1, half-1:-1] = Const.FLOOR
                case _:
                    tilemap[half-3:half+4, half-3:half+4] = Const.FLOOR
        case Shape.HALF:
            match room_val & 0b01111:
                case 0b01110:
                    tilemap[half:-1, 1:-1] = Const.FLOOR
                case 0b01101:
                    tilemap[1:-1, 1:half] = Const.FLOOR
                case 0b01011:
                    tilemap[1:half, 1:-1] = Const.FLOOR
                case 0b00111:
                    tilemap[1:-1, half:-1] = Const.FLOOR
                case _:
                    pass
    return tilemap

@timeit
def get_theme(room_shape: Shape, rand_rng: Random) -> Theme:
    """
    Randomly decides room theme dependent on room shape

    Parameters
    ----------
    shape : str
        shape of room
    np_rng : np.random.Generator

    Returns
    -------
    theme : str
        theme of room
    """
    if room_shape not in _THEME_TABLES:
        raise InvalidRoom(f"The get_theme function does not support rooms with {room_shape} shape")
    theme_list, probs = _THEME_TABLES[room_shape]
    theme: Theme = rand_rng.choices(theme_list, probs)[0]
    return theme

@timeit
def scan_tilemap(tilemap: array[uint8], neighbor_map: array[uint8] | None = None, require: Collection[int] | None = None,
                 block: Collection[int] | None = None, bias: Collection[int] | None = None,
                 place_on: Collection[int] | None = None) -> array[np.int32]:
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
    if place_on is None: place_on = {Const.FLOOR}

    available_grid = np.isin(tilemap,tuple(place_on))
    neighbor_map = np.empty_like(tilemap, dtype = uint8)

    if require is not None:
        available_grid &= (adj_map(tilemap, neighbor_map, target = require, iso = False) != 0)
    if block is not None:
        available_grid &= (adj_map(tilemap, neighbor_map, target = block, iso = False) == 0)
    
    available_list = np.argwhere(available_grid)
    if bias is not None:
        bias_grid = available_grid & (adj_map(tilemap, neighbor_map, target = bias, iso = False) != 0)
        bias_mask = bias_grid[available_list[:,0], available_list[:, 1]]
        biases = available_list[bias_mask]
        if biases.size > 0:
            bias_list = np.repeat(biases, 4, axis=0)
            available_list = np.concatenate((available_list,bias_list),axis = 0)
    return available_list

@timeit
def populate_tilemap(tilemap: array[uint8], theme: Theme, np_rng: np.random.Generator) -> array[uint8]:
    """
    Populates tilemap with features

    Parameters
    ----------
    tilemap : NDArray[uint8]
        tilemap to populate
    theme : str
        theme of room
    np_rng : np.random.Generator
        seeded random

    Returns
    -------
    tilemap : NDArray[uint8]
        populated tilemap
    """
    feature_order = (
        Const.WATER,
        Const.HOLE,
        Const.HEALING_STATION,
        Const.SHRINE,
        Const.CHEST,
        Const.LOOT_PILE,
        Const.TRAP,
        Const.BOSS_SPAWNER,
        Const.MONSTER_SPAWNER
    )
    def R(num: int = 0) -> int: return np_rng.integers(0,1)+num
    def T(num: int = 0) -> int: return np_rng.integers(0,2)+num
    population_dict: dict[Theme, dict[Const, int]] = {
        Theme.DE_TRAPPED:  {Const.HOLE: 1, Const.WATER: R(), Const.TRAP: 3},
        Theme.DE_TREASURE: {Const.TRAP: 1, Const.CHEST: 1, Const.LOOT_PILE: 2, Const.MONSTER_SPAWNER: 1},
        Theme.DE_HEALTHY:  {Const.HEALING_STATION: 1},
        Theme.DE_GUARDED:  {Const.MONSTER_SPAWNER: 1},

        Theme.SR_TRAPPED:  {Const.HOLE: 1, Const.TRAP: T(3), Const.LOOT_PILE: 1, Const.MONSTER_SPAWNER: 1},
        Theme.SR_TREASURE: {Const.TRAP: R(1), Const.CHEST: 2, Const.LOOT_PILE: 3},
        Theme.SR_GUARDED:  {Const.WATER: R(), Const.TRAP: 1, Const.LOOT_PILE: 1, Const.MONSTER_SPAWNER: 2},
        Theme.SR_CHAOS:    {Const.HOLE: 2, Const.WATER: R(), Const.TRAP: 3, Const.CHEST: 1, Const.LOOT_PILE: 2, Const.MONSTER_SPAWNER: 3, Const.SHRINE: 1},
        Theme.SR_BASIC:    {Const.TRAP: R(), Const.LOOT_PILE: R()},
        
        Theme.CN_TRAPPED:  {Const.HOLE: 1, Const.TRAP: T(1), Const.LOOT_PILE: 1},
        Theme.CN_GUARDED:  {Const.MONSTER_SPAWNER: 1},
        Theme.CN_BASIC:    {Const.LOOT_PILE: R()},
        
        Theme.LR_TRAPPED:  {Const.HOLE: 2, Const.WATER: 1, Const.TRAP: T(3), Const.LOOT_PILE: 2, Const.MONSTER_SPAWNER: 1},
        Theme.LR_TREASURE: {Const.TRAP: 1, Const.CHEST: 2, Const.LOOT_PILE: 3, Const.MONSTER_SPAWNER: 1},
        Theme.LR_HEALTHY:  {Const.HEALING_STATION: 1},
        Theme.LR_GUARDED:  {Const.WATER: R(), Const.TRAP: 1, Const.CHEST: 1, Const.LOOT_PILE: 1, Const.MONSTER_SPAWNER: 3},
        Theme.LR_CHAOS:    {Const.HOLE: 2, Const.WATER: 1, Const.TRAP: 3, Const.CHEST: 2, Const.LOOT_PILE: 3, Const.MONSTER_SPAWNER: T(2), Const.SHRINE: 1},
        Theme.LR_BASIC:    {Const.TRAP: R(1), Const.LOOT_PILE: R()},
        
        Theme.CR_TRAPPED:  {Const.HOLE: 1, Const.TRAP: T(2), Const.LOOT_PILE: 1},
        Theme.CR_TREASURE: {Const.TRAP: 1, Const.CHEST: 1, Const.LOOT_PILE: 3, Const.MONSTER_SPAWNER: 1},
        Theme.CR_GUARDED:  {Const.WATER: R(), Const.TRAP: 1, Const.LOOT_PILE: 1, Const.MONSTER_SPAWNER: 2},
        Theme.CR_CHAOS:    {Const.HOLE: R(), Const.WATER: 1, Const.TRAP: 3, Const.CHEST: T(), Const.LOOT_PILE: 3, Const.MONSTER_SPAWNER: R(2), Const.SHRINE: 1},
        Theme.CR_BASIC:    {Const.TRAP: R(), Const.LOOT_PILE: R()},
        
        Theme.HR_TRAPPED:  {Const.HOLE: 1, Const.TRAP: T(2), Const.LOOT_PILE: 1},
        Theme.HR_TREASURE: {Const.TRAP: 1, Const.CHEST: 1, Const.LOOT_PILE: 3, Const.MONSTER_SPAWNER: 1},
        Theme.HR_GUARDED:  {Const.WATER: R(), Const.TRAP: 1, Const.LOOT_PILE: 1, Const.MONSTER_SPAWNER: 2},
        Theme.HR_CHAOS:    {Const.HOLE: R(), Const.WATER: 1, Const.TRAP: 3, Const.MONSTER_SPAWNER: R(2), Const.SHRINE: 1, Const.CHEST: T(), Const.LOOT_PILE: 3},
        Theme.HR_BASIC:    {Const.TRAP: R(), Const.LOOT_PILE: R()},
        
        Theme.BR_HOARD:    {Const.CHEST: 3, Const.LOOT_PILE: 9, Const.BOSS_SPAWNER: 1, Const.SHRINE: 1},
        Theme.BR_WIZARD:   {Const.CHEST: 4, Const.LOOT_PILE: 3, Const.BOSS_SPAWNER: 1, Const.SHRINE: 1},
        Theme.BR_WEAK:     {Const.TRAP: R(), Const.CHEST: 1, Const.LOOT_PILE: 2, Const.MONSTER_SPAWNER: 1, Const.BOSS_SPAWNER: 1},
        Theme.BR_STRONG:   {Const.HEALING_STATION: R(), Const.CHEST: T(1), Const.LOOT_PILE: 5, Const.BOSS_SPAWNER: 1, Const.SHRINE: 1},
        Theme.BR_GUARDED:  {Const.TRAP: 1, Const.CHEST: 2, Const.LOOT_PILE: 3, Const.MONSTER_SPAWNER: 2, Const.BOSS_SPAWNER: 1, Const.SHRINE: 1},
        Theme.BR_DOUBLE:   {Const.CHEST: 3, Const.LOOT_PILE: 5, Const.BOSS_SPAWNER: 2, Const.SHRINE: 1},
        
        Theme.EMPTY:       {}
    }
    pop_vals = population_dict[theme]
    for feature in feature_order:
        count = pop_vals.get(feature)
        if count:
            match feature:
                case Const.HOLE:
                    available_list = scan_tilemap(tilemap, block = {Const.WALL, Const.WATER, Const.LOOT_PILE})
                    indices = np_rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = Const.HOLE
                case Const.WATER:
                    available_list = scan_tilemap(tilemap, block = {Const.CHEST, Const.LOOT_PILE, Const.HOLE})
                    indices = np_rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = Const.WATER
                case Const.TRAP:
                    available_list = scan_tilemap(tilemap, block = {Const.TRAP, Const.HEALING_STATION, Const.SHRINE})
                    indices = np_rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = Const.TRAP
                case Const.HEALING_STATION:
                    available_list = scan_tilemap(tilemap, require = {Const.FLOOR}, place_on = {Const.WALL})
                    indices = np_rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = Const.HEALING_STATION
                case Const.CHEST:
                    available_list = scan_tilemap(tilemap, bias = {Const.LOOT_PILE, Const.WALL})
                    indices = np_rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = Const.CHEST
                case Const.LOOT_PILE:
                    available_list = scan_tilemap(tilemap, bias = {Const.CHEST, Const.LOOT_PILE}, block = {Const.WATER, Const.HOLE})
                    for _ in range(count):
                        if len(available_list) > 0:
                            i = np_rng.integers(0, len(available_list))
                            row, col = available_list[i]
                            tilemap[row,col] = Const.LOOT_PILE
                            available_list = np.delete(available_list, i, axis=0).astype(np.int32)
                case Const.MONSTER_SPAWNER:
                    available_list = scan_tilemap(tilemap, block = {Const.BOSS_SPAWNER, Const.HEALING_STATION, Const.SHRINE})
                    indices = np_rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = Const.MONSTER_SPAWNER
                case Const.BOSS_SPAWNER:
                    available_list = scan_tilemap(tilemap, block = {Const.MONSTER_SPAWNER, Const.HEALING_STATION, Const.SHRINE})
                    indices = np_rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = Const.BOSS_SPAWNER
                case Const.SHRINE:
                    available_list = scan_tilemap(tilemap, require = {Const.FLOOR}, place_on = {Const.WALL})
                    indices = np_rng.choice(len(available_list), size=count, replace=False)
                    coords = available_list[indices]
                    rows = coords[:, 0]
                    cols = coords[:, 1]
                    tilemap[rows,cols] = Const.SHRINE
    return tilemap

@timeit
def room_map_generator(room_val: int, np_rng: np.random.Generator, rand_rng: Random) -> tuple[array[uint8], Shape, Theme]:
    """
    Handler function to create Room map

    Parameters
    ----------
    room_val : int
        value of room tile to generate
    np_rng : np.random.Generator
        numpy seeded np_rng

    Returns
    -------
    tilemap : NDArray[uint8]
        final room tilemap
    shape : str
        shape of room
    theme : str
        theme of room
    """
    tilemap = init_tilemap(Const.ROOM_SIZE)
    shape = get_shape(room_val, rand_rng)
    tilemap = build_room(tilemap, room_val, shape, np_rng)
    theme = get_theme(shape, rand_rng)
    tilemap = populate_tilemap(tilemap, theme, np_rng)
    return tilemap, shape, theme

#region DEBUG
def _on_click(event: Event, ax: Axes, tilemap: array[uint8], room_shape: Shape, room_theme: Theme) -> None:
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
            print(f"\033cShape: {room_shape}\nTheme: {room_theme}\n"+
                f"Tile Clicked: {row}, {col}\n"+
                f"Tile Value: {Const(tilemap[row,col]).name}")
    else:
        print(f"\033cShape: {room_shape}\nTheme: {room_theme}")
    return

def _debug(tilemap: array[uint8], room_shape: Shape, room_theme: Theme) -> None:
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
                           _on_click(event,ax,tilemap,room_shape,room_theme))

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
        room_val = np_rng.integers(17, 31, endpoint = True)
        start = clock()
        _ = room_map_generator(room_val, np_rng, rand_rng)
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
    rand_rng = Random(debug_seed)

    tilemap, shape, theme = room_map_generator(debug_room_val, np_rng, rand_rng)

    print(f"\033cShape: {shape}\nTheme: {theme}\n")
    _debug(tilemap, shape, theme)
    return
#endregion

if __name__ == "__main__":
    if arg_parser(): _time_test(int(input("Input Testing Count: ")))
    else: _main()