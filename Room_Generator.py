"""
**Room Map Generator**

In File Entry Point: _main() or _time_test()
Import Entry Point: room_map_generator()
"""

import numpy as np

from numpy import uint8
from numpy.typing import NDArray as array
from collections.abc import Collection
from time import perf_counter_ns as clock
from enum import IntEnum
from random import Random

from Generator_Helpers import init_tilemap, adj_map
from Debug_Tools import timeit, arg_parser, debug_render

class InvalidRoom(Exception):
    """
    Invalid Room Error for manual calls of functions
    Program will never throw this error on it's own
    """
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
    LOOT_CLUSTER = 7
    MONSTER_SPAWNER = 8
    BOSS_SPAWNER = 9
    SHRINE = 10

class Shape(IntEnum):
    """
    Shape Constants for Rooms
    """
    DEAD_END = 1
    BOSS_ROOM = 2
    SMALL_ROOM = 3
    LARGE_ROOM = 4
    CONNECTION = 5
    CORNER = 6
    HALF = 7

class Theme(IntEnum):
    """
    Theme Constants for Rooms
    """
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

_POPULATION_TABLES: dict[Theme, dict[Const, int | tuple[int, int]]] = {
    Theme.DE_TRAPPED:  {Const.HOLE: 1, Const.WATER: (0,1), Const.TRAP: 3},
    Theme.DE_TREASURE: {Const.TRAP: 1, Const.CHEST: 1, Const.LOOT_PILE: 2, Const.MONSTER_SPAWNER: 1},
    Theme.DE_HEALTHY:  {Const.HEALING_STATION: 1},
    Theme.DE_GUARDED:  {Const.MONSTER_SPAWNER: 1},

    Theme.SR_TRAPPED:  {Const.HOLE: 1, Const.TRAP: (3,5), Const.LOOT_PILE: 1, Const.MONSTER_SPAWNER: 1},
    Theme.SR_TREASURE: {Const.TRAP: (1,2), Const.CHEST: 2, Const.LOOT_PILE: 3},
    Theme.SR_GUARDED:  {Const.WATER: (0,1), Const.TRAP: 1, Const.LOOT_PILE: 1, Const.MONSTER_SPAWNER: 2},
    Theme.SR_CHAOS:    {Const.HOLE: 2, Const.WATER: (0,1), Const.TRAP: 3, Const.CHEST: 1, Const.LOOT_PILE: 2, Const.MONSTER_SPAWNER: 3, Const.SHRINE: 1},
    Theme.SR_BASIC:    {Const.TRAP: (0,1), Const.LOOT_PILE: (0,1)},
    
    Theme.CN_TRAPPED:  {Const.HOLE: 1, Const.TRAP: (1,3), Const.LOOT_PILE: 1},
    Theme.CN_GUARDED:  {Const.MONSTER_SPAWNER: 1},
    Theme.CN_BASIC:    {Const.LOOT_PILE: (0,1)},
    
    Theme.LR_TRAPPED:  {Const.HOLE: 2, Const.WATER: 1, Const.TRAP: (3,5), Const.LOOT_PILE: 2, Const.MONSTER_SPAWNER: 1},
    Theme.LR_TREASURE: {Const.TRAP: 1, Const.CHEST: 2, Const.LOOT_PILE: 3, Const.MONSTER_SPAWNER: 1},
    Theme.LR_HEALTHY:  {Const.HEALING_STATION: 1},
    Theme.LR_GUARDED:  {Const.WATER: (0,1), Const.TRAP: 1, Const.CHEST: 1, Const.LOOT_PILE: 1, Const.MONSTER_SPAWNER: 3},
    Theme.LR_CHAOS:    {Const.HOLE: 2, Const.WATER: 1, Const.TRAP: 3, Const.CHEST: 2, Const.LOOT_PILE: 3, Const.MONSTER_SPAWNER: (2,4), Const.SHRINE: 1},
    Theme.LR_BASIC:    {Const.TRAP: (1,2), Const.LOOT_PILE: (0,1)},
    
    Theme.CR_TRAPPED:  {Const.HOLE: 1, Const.TRAP: (2,4), Const.LOOT_PILE: 1},
    Theme.CR_TREASURE: {Const.TRAP: 1, Const.CHEST: 1, Const.LOOT_PILE: 3, Const.MONSTER_SPAWNER: 1},
    Theme.CR_GUARDED:  {Const.WATER: (0,1), Const.TRAP: 1, Const.LOOT_PILE: 1, Const.MONSTER_SPAWNER: 2},
    Theme.CR_CHAOS:    {Const.HOLE: (0,1), Const.WATER: 1, Const.TRAP: 3, Const.CHEST: (0,2), Const.LOOT_PILE: 3, Const.MONSTER_SPAWNER: (2,3), Const.SHRINE: 1},
    Theme.CR_BASIC:    {Const.TRAP: (0,1), Const.LOOT_PILE: (0,1)},
    
    Theme.HR_TRAPPED:  {Const.HOLE: 1, Const.TRAP: (2,4), Const.LOOT_PILE: 1},
    Theme.HR_TREASURE: {Const.TRAP: 1, Const.CHEST: 1, Const.LOOT_PILE: 3, Const.MONSTER_SPAWNER: 1},
    Theme.HR_GUARDED:  {Const.WATER: (0,1), Const.TRAP: 1, Const.LOOT_PILE: 1, Const.MONSTER_SPAWNER: 2},
    Theme.HR_CHAOS:    {Const.HOLE: (0,1), Const.WATER: 1, Const.TRAP: 3, Const.MONSTER_SPAWNER: (2,3), Const.SHRINE: 1, Const.CHEST: (0,2), Const.LOOT_PILE: 3},
    Theme.HR_BASIC:    {Const.TRAP: (0,1), Const.LOOT_PILE: (0,1)},
    
    Theme.BR_HOARD:    {Const.CHEST: 3, Const.LOOT_PILE: 9, Const.BOSS_SPAWNER: 1, Const.SHRINE: 1},
    Theme.BR_WIZARD:   {Const.CHEST: 4, Const.LOOT_PILE: 3, Const.BOSS_SPAWNER: 1, Const.SHRINE: 1},
    Theme.BR_WEAK:     {Const.TRAP: (0,1), Const.CHEST: 1, Const.LOOT_PILE: 2, Const.MONSTER_SPAWNER: 1, Const.BOSS_SPAWNER: 1},
    Theme.BR_STRONG:   {Const.HEALING_STATION: (0,1), Const.CHEST: (1,3), Const.LOOT_PILE: 5, Const.BOSS_SPAWNER: 1, Const.SHRINE: 1},
    Theme.BR_GUARDED:  {Const.TRAP: 1, Const.CHEST: 2, Const.LOOT_PILE: 3, Const.MONSTER_SPAWNER: 2, Const.BOSS_SPAWNER: 1, Const.SHRINE: 1},
    Theme.BR_DOUBLE:   {Const.CHEST: 3, Const.LOOT_PILE: 5, Const.BOSS_SPAWNER: 2, Const.SHRINE: 1},
    
    Theme.EMPTY:       {}
    }

_FEATURE_ORDER = (
        Const.WATER,
        Const.HOLE,
        Const.HEALING_STATION,
        Const.SHRINE,
        Const.CHEST,
        Const.LOOT_PILE,
        Const.LOOT_CLUSTER,
        Const.TRAP,
        Const.BOSS_SPAWNER,
        Const.MONSTER_SPAWNER
    )

_SCAN_PARAMS: dict[Const, dict[str, set[Const]]] = {
    Const.WATER:            {"block": {Const.CHEST, Const.LOOT_PILE, Const.HOLE}},
    Const.HOLE:             {"block": {Const.WALL, Const.WATER, Const.LOOT_PILE}},
    Const.HEALING_STATION:  {"require": {Const.FLOOR}, "place_on": {Const.WALL}},
    Const.SHRINE:           {"require": {Const.FLOOR}, "place_on": {Const.WALL}},
    Const.CHEST:            {"bias": {Const.LOOT_PILE, Const.WALL}},
    Const.LOOT_PILE:        {"bias": {Const.CHEST}, "block": {Const.WATER, Const.HOLE}},
    Const.LOOT_CLUSTER:     {"require": {Const.CHEST, Const.LOOT_PILE}, "block": {Const.WATER, Const.HOLE}},
    Const.TRAP:             {"block": {Const.TRAP, Const.HEALING_STATION, Const.SHRINE}},
    Const.BOSS_SPAWNER:     {"block": {Const.MONSTER_SPAWNER, Const.HEALING_STATION, Const.SHRINE}},
    Const.MONSTER_SPAWNER:  {"block": {Const.BOSS_SPAWNER, Const.HEALING_STATION, Const.SHRINE}}
    }

@timeit
def get_shape(room_val: int, rand_rng: Random) -> Shape:
    """
    Randomly selects a room shape based on the number of exits encoded
    in the room's tile value.

    Parameters
    ----------
    room_val : int
        Dungeon tile value. Must be in the range 0b10000 to 0b11111
        (bit 4 set, bits 0-3 representing exits).
    rand_rng : Random
        Python random generator used for weighted shape selection.

    Returns
    -------
    room_shape : Shape
        Randomly selected shape for the room.

    Raises
    ------
    InvalidRoom
        If room_val is outside the valid range, or if the number of
        exits encoded in bits 0-3 is not supported.

    Notes
    -----
    - Bits 0-3 of room_val encode the orthogonal exits:

        Bit 0 (value 1) : North
        Bit 1 (value 2) : East
        Bit 2 (value 4) : South
        Bit 3 (value 8) : West

    - The exit count is used to look up a weighted shape list in
      _SHAPE_TABLES, from which a shape is randomly selected.
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
    Builds room exits and central shape onto a blank tilemap in two passes.

    Parameters
    ----------
    tilemap : array[uint8]
        Blank 2D array of size Const.ROOM_SIZE x Const.ROOM_SIZE.
    room_val : int
        Dungeon tile value. Bits 0-3 encode orthogonal exits:

            Bit 0 (value 1) : North
            Bit 1 (value 2) : East
            Bit 2 (value 4) : South
            Bit 3 (value 8) : West

    room_shape : Shape
        Shape of the room, determining how the central area is filled.
    np_rng : np.random.Generator
        NumPy random generator, used for randomising Dead End size.

    Returns
    -------
    tilemap : array[uint8]
        Tilemap with exits and room shape written in.

    Notes
    -----
    Pass 1 - Exits:
        Each active bit in room_val carves a 3-tile wide corridor from
        the edge of the tilemap to the midpoint, ensuring rooms align
        correctly when placed adjacent to each other in the dungeon.

    Pass 2 - Shape:
        DEAD_END   : Random square centered on midpoint, 2-5 tiles out.
                     Note: exits are carved first, then the center is
                     reset to Const.WALL, creating a dead end pocket.
        BOSS_ROOM  : Fills almost the entire tilemap wall to wall.
        SMALL_ROOM : Fixed 7x7 square centered on midpoint.
        CONNECTION : Minimal 3x3 square, just enough to join corridors.
        LARGE_ROOM : Fixed 13x13 square centered on midpoint.
        CORNER     : Fills the quadrant shared by the two active exits.
                     Falls back to a small room for unexpected combinations.
        HALF       : Fills the half of the room opposite the missing exit.
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
    Randomly selects a room theme based on the room's shape.

    Parameters
    ----------
    room_shape : Shape
        Shape of the room, used to look up the weighted theme list
        in _THEME_TABLES.
    rand_rng : Random
        Python random generator used for weighted theme selection.

    Returns
    -------
    theme : Theme
        Randomly selected theme for the room.

    Raises
    ------
    InvalidRoom
        If room_shape is not present in _THEME_TABLES.
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
    Scans a tilemap and returns a list of valid tile indices to place on.

    Parameters
    ----------
    tilemap : array[uint8]
        Tilemap to scan.
    neighbor_map : array[uint8] | None, optional
        Pre-allocated buffer for adjacency calculations. If not provided,
        a new buffer is allocated. Pass a pre-allocated buffer when calling
        scan_tilemap multiple times to avoid repeated allocation overhead.
    require : Collection[int] | None, optional
        If provided, only tiles orthogonally adjacent to a tile in this
        collection are considered valid placement targets.
    block : Collection[int] | None, optional
        If provided, tiles orthogonally adjacent to a tile in this
        collection are excluded from valid placement targets.
    bias : Collection[int] | None, optional
        If provided, tiles orthogonally adjacent to a tile in this
        collection are repeated 4 extra times in the returned list,
        increasing their probability of selection.
    place_on : Collection[int] | None, optional
        If provided, only tiles whose values are in this collection are
        considered as placement targets. Defaults to Const.FLOOR if not
        provided, using a fast equality check instead of np.isin.

    Returns
    -------
    available_list : array[np.int32]
        2D array of [row, col] indices representing valid placement targets.
        Biased tiles appear multiple times to reflect their higher weight.
    """
    if place_on is None: available_grid = tilemap == Const.FLOOR
    else: available_grid = np.isin(tilemap,tuple(place_on))

    if neighbor_map is None: neighbor_map = np.empty_like(tilemap, dtype=uint8)

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
def place(tilemap: array[uint8], feature: Const, available_list: array[np.int32], count: int, np_rng: np.random.Generator) -> None:
    """
    Places a feature onto randomly selected tiles in the tilemap.

    Parameters
    ----------
    tilemap : array[uint8]
        Tilemap to place the feature onto.
    feature : Const
        Tile value to write at each selected position.
    available_list : array[np.int32]
        2D array of [row, col] indices representing valid placement targets,
        as returned by scan_tilemap.
    count : int
        Number of tiles to place. Clamped to len(available_list) if the
        available list is smaller than the requested count.
    np_rng : np.random.Generator
        NumPy random generator used for index selection.

    Returns
    -------
    None

    Notes
    -----
    - Placement is performed without replacement, so no tile will be
    selected more than once per call.
    - If available_list is empty, the function returns immediately
    without placing anything.
    """
    count = min(count, len(available_list))
    if not count: return
    indices = np_rng.choice(len(available_list), size=count, replace=False)
    coords = available_list[indices]
    tilemap[coords[:, 0], coords[:, 1]] = feature
    return

@timeit
def populate_tilemap(tilemap: array[uint8], theme: Theme, np_rng: np.random.Generator) -> array[uint8]:
    """
    Populates a room tilemap with features based on the given theme.

    Parameters
    ----------
    tilemap : array[uint8]
        Tilemap to populate, with room shape already built.
    theme : Theme
        Theme of the room, used to look up feature counts in
        _POPULATION_TABLES.
    np_rng : np.random.Generator
        NumPy random generator used for feature count resolution
        and placement selection.

    Returns
    -------
    tilemap : array[uint8]
        Tilemap with features placed.

    Notes
    -----
    - Feature counts are resolved from _POPULATION_TABLES at the start
      of each call. Fixed counts are used directly, while tuple entries
      are resolved to a random integer within the given range.
    - Features are placed in the order defined by _FEATURE_ORDER, ensuring
      that earlier features are present when later features scan for
      adjacency requirements or blocks.
    - A single neighbor_map buffer is pre-allocated and shared across all
      scan_tilemap calls to avoid repeated allocation overhead.
    - Placement parameters for each feature are looked up from _SCAN_PARAMS.
    - If a theme does not include a feature, or the resolved count is 0,
      that feature is skipped entirely.
    """
    pop_vals: dict[Const, int] = {}
    for feature, count in _POPULATION_TABLES[theme].items():
        if isinstance(count, tuple):
            pop_vals[feature] = np_rng.integers(count[0], count[1], endpoint=True)
        else:
            pop_vals[feature] = count
    neighbor_map = np.empty_like(tilemap, dtype = uint8)
    for feature in _FEATURE_ORDER:
        if feature not in pop_vals: continue
        count = pop_vals[feature]
        if not count: continue
        available_list = scan_tilemap(tilemap, neighbor_map, **_SCAN_PARAMS[feature])
        place(tilemap, feature, available_list, count, np_rng)
    return tilemap

@timeit
def room_map_generator(room_val: int, np_rng: np.random.Generator, rand_rng: Random) -> tuple[array[uint8], Shape, Theme]:
    """
    Generates a complete room tilemap for a given dungeon tile value.

    Parameters
    ----------
    room_val : int
        Dungeon tile value representing the room to generate. Must be
        in the range 0b10000 to 0b11111.
    np_rng : np.random.Generator
        NumPy random generator used for shape building and feature placement.
    rand_rng : Random
        Python random generator used for shape and theme selection.

    Returns
    -------
    tilemap : array[uint8]
        Final populated room tilemap.
    shape : Shape
        Shape of the generated room.
    theme : Theme
        Theme of the generated room.

    Notes
    -----
    Generation pipeline:
        1. init_tilemap()      : Allocates a blank ROOM_SIZE x ROOM_SIZE array.
        2. get_shape()         : Selects a weighted random shape based on exit count.
        3. build_room()        : Carves exits and fills the central room area.
        4. get_theme()         : Selects a weighted random theme based on shape.
        5. populate_tilemap()  : Places features according to the selected theme.
    """
    tilemap = init_tilemap(Const.ROOM_SIZE)
    shape = get_shape(room_val, rand_rng)
    tilemap = build_room(tilemap, room_val, shape, np_rng)
    theme = get_theme(shape, rand_rng)
    tilemap = populate_tilemap(tilemap, theme, np_rng)
    return tilemap, shape, theme

#region DEBUG
def _get_tile_value(value: int) -> tuple[str,str]:
    return ("Tile: ", Const(value).name)

def _debug(tilemap: array[uint8], room_shape: Shape, room_theme: Theme) -> None:
    colours = ["black", "white", "gray", "blue", "red", "green", "brown", "yellow", "orange", "red", "green"]
    info = {"Shape": room_shape.name, "Theme": room_theme.name}
    debug_render(tilemap, colours, info, tile_formatter = _get_tile_value)
    return

def _time_test(count: int) -> None:
    """
    Runs the room map generator a specified number of times and reports
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
    - A random room_val in the range 17-31 is generated for each run,
      covering all valid single-active-tile room values.
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
    Debug entry point to the program. Prompts for a room value and optional
    seed, generates a room, and launches the debug visualizer.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    - The console is cleared at the start using the '\\033c' escape code.
    - The user is prompted for a room value and an optional integer seed:
        - If a seed is provided, both random generators are initialized
          with it, making the run fully reproducible.
        - If no seed is entered, both generators are initialized without
          a seed, producing a random result each run.
    - The room shape and theme names are printed to the console before
      the debug visualizer is launched.
    - This is an internal entry point.
    """
    print("\033c", end="")

    debug_room_val = int(input("Input Room Value: "))
    user_input = input("Input Seed: ")
    debug_seed = int(user_input) if user_input else None
    np_rng = np.random.default_rng(debug_seed)
    rand_rng = Random(debug_seed)

    tilemap, shape, theme = room_map_generator(debug_room_val, np_rng, rand_rng)

    print(f"\033cShape: {shape.name}\nTheme: {theme.name}\n")
    _debug(tilemap, shape, theme)
    return
#endregion

if __name__ == "__main__":
    if arg_parser(): _time_test(int(input("Input Testing Count: ")))
    else: _main()