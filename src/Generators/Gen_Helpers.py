import numpy as np

from enum import IntEnum, auto

class Shape(IntEnum):
    DEAD_END = 0
    BOSS_ROOM = auto()
    SMALL_ROOM = auto()
    LARGE_ROOM = auto()
    CONNECTION = auto()
    CORNER = auto()
    HALF = auto()
    SMALL_CIRCLE = auto()
    LARGE_CIRCLE = auto()

class Theme(IntEnum):
    NULL = 0
    EMPTY = auto()
    ENTRANCE = auto()

    DE_TRAPPED = auto()
    DE_TREASURE = auto()
    DE_HEALTHY = auto()
    DE_GUARDED = auto()

    BR_HOARD = auto()
    BR_WIZARD = auto()
    BR_WEAK = auto()
    BR_STRONG = auto()
    BR_GUARDED = auto()
    BR_DOUBLE = auto()

    SR_TRAPPED = auto()
    SR_TREASURE = auto()
    SR_GUARDED = auto()
    SR_CHAOS = auto()
    SR_BASIC = auto()
    SR_FLOODED = auto()

    CN_TRAPPED = auto()
    CN_GUARDED = auto()
    CN_BASIC = auto()
    CN_FLOODED = auto()

    LR_TRAPPED = auto()
    LR_TREASURE = auto()
    LR_HEALTHY = auto()
    LR_GUARDED = auto()
    LR_CHAOS = auto()
    LR_BASIC = auto()
    LR_FLOODED = auto()

    CR_TRAPPED = auto()
    CR_TREASURE = auto()
    CR_GUARDED = auto()
    CR_CHAOS = auto()
    CR_BASIC = auto()
    CR_FLOODED = auto()

    HR_TRAPPED = auto()
    HR_TREASURE = auto()
    HR_GUARDED = auto()
    HR_CHAOS = auto()
    HR_BASIC = auto()
    HR_FLOODED = auto()

    SC_TRAPPED = auto()
    SC_TREASURE = auto()
    SC_GUARDED = auto()
    SC_CHAOS = auto()
    SC_BASIC = auto()
    SC_FLOODED = auto()
    
    LC_TRAPPED = auto()
    LC_TREASURE = auto()
    LC_HEALTHY = auto()
    LC_GUARDED = auto()
    LC_CHAOS = auto()
    LC_BASIC = auto()
    LC_FLOODED = auto()

class Const(IntEnum):
    ROOM_SIZE = 17
    HALF = ROOM_SIZE//2
    WALL = 0
    FLOOR = 1
    HOLE = 2
    WATER = 3
    WATER_POOL = 4
    TRAP = 5
    HEALING_STATION = 6
    CHEST = 7
    LOOT_PILE = 8
    LOOT_CLUSTER = 9
    MONSTER_SPAWNER = 10
    BOSS_SPAWNER = 11
    SHRINE = 12
    PAINT_1 = 13
    PAINT_2 = 14
    PAINT_3 = 15
    ENTRANCE = 16

class S1_Const(IntEnum):
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

class InvalidRoom(Exception):
    pass

SHAPE_TABLES: dict[int, tuple[list[Shape], list[float]]] = {
    1: (
        [Shape.DEAD_END, Shape.BOSS_ROOM, Shape.SMALL_ROOM, Shape.SMALL_CIRCLE, Shape.LARGE_CIRCLE],
        [0.30, 0.10, 0.35, 0.15, 0.10]
        ),
    2: (
        [Shape.CONNECTION, Shape.SMALL_ROOM, Shape.LARGE_ROOM, Shape.CORNER, Shape.SMALL_CIRCLE],
        [0.10, 0.25, 0.30, 0.20, 0.15]
        ),
    3: (
        [Shape.CONNECTION, Shape.SMALL_ROOM, Shape.LARGE_ROOM, Shape.HALF, Shape.SMALL_CIRCLE, Shape.LARGE_CIRCLE],
        [0.15, 0.20, 0.28, 0.17, 0.12, 0.08]
        ),
    4: (
        [Shape.CONNECTION, Shape.SMALL_ROOM, Shape.LARGE_ROOM, Shape.LARGE_CIRCLE],
        [0.15, 0.28, 0.40, 0.17]
        )
}

THEME_TABLES: dict[Shape, tuple[list[Theme], list[float]]] = {
    Shape.DEAD_END:
    (
        [Theme.DE_TRAPPED, Theme.DE_TREASURE, Theme.DE_HEALTHY, Theme.DE_GUARDED, Theme.EMPTY],
        [0.20, 0.15, 0.10, 0.15, 0.40]
    ),
    Shape.BOSS_ROOM:
    (
        [Theme.BR_HOARD, Theme.BR_WIZARD, Theme.BR_WEAK, Theme.BR_STRONG, Theme.BR_GUARDED, Theme.BR_DOUBLE],
        [0.20, 0.20, 0.20, 0.10, 0.20, 0.10]
    ),
    Shape.SMALL_ROOM:
    (
        [Theme.SR_TRAPPED, Theme.SR_TREASURE, Theme.SR_GUARDED, Theme.SR_CHAOS, Theme.SR_BASIC, Theme.SR_FLOODED, Theme.EMPTY],
        [0.20, 0.10, 0.15, 0.10, 0.25, 0.10, 0.10]
    ),
    Shape.CONNECTION:
    (
        [Theme.CN_TRAPPED, Theme.CN_GUARDED, Theme.CN_BASIC, Theme.CN_FLOODED, Theme.EMPTY],
        [0.20, 0.20, 0.25, 0.10, 0.25]
    ),
    Shape.LARGE_ROOM:
    (
        [Theme.LR_TRAPPED, Theme.LR_TREASURE, Theme.LR_HEALTHY, Theme.LR_GUARDED, Theme.LR_CHAOS, Theme.LR_BASIC, Theme.LR_FLOODED, Theme.EMPTY],
        [0.20, 0.05, 0.05, 0.15, 0.10, 0.25, 0.10, 0.10]
    ),
    Shape.CORNER:
    (
        [Theme.CR_TRAPPED, Theme.CR_TREASURE, Theme.CR_GUARDED, Theme.CR_CHAOS, Theme.CR_BASIC, Theme.CR_FLOODED, Theme.EMPTY],
        [0.20, 0.10, 0.15, 0.10, 0.25, 0.10, 0.10]
    ),
    Shape.HALF:
    (
        [Theme.HR_TRAPPED, Theme.HR_TREASURE, Theme.HR_GUARDED, Theme.HR_CHAOS, Theme.HR_BASIC, Theme.HR_FLOODED, Theme.EMPTY],
        [0.20, 0.10, 0.15, 0.10, 0.25, 0.10, 0.10]
    ),
    Shape.SMALL_CIRCLE:
    (
        [Theme.SC_TRAPPED, Theme.SC_TREASURE, Theme.SC_GUARDED, Theme.SC_CHAOS, Theme.SC_BASIC, Theme.SC_FLOODED, Theme.EMPTY],
        [0.20, 0.10, 0.15, 0.10, 0.25, 0.10, 0.10]
    ),
    Shape.LARGE_CIRCLE:
    (
        [Theme.LC_TRAPPED, Theme.LC_TREASURE, Theme.LC_HEALTHY, Theme.LC_GUARDED, Theme.LC_CHAOS, Theme.LC_BASIC, Theme.LC_FLOODED, Theme.EMPTY],
        [0.20, 0.05, 0.05, 0.15, 0.10, 0.25, 0.10, 0.10]
    ),
}

POPULATION_TABLES: dict[Theme, dict[Const, int | tuple[int, int]]] = {
    Theme.DE_TRAPPED:  {Const.HOLE: 1, Const.WATER: (0,1), Const.TRAP: 3},
    Theme.DE_TREASURE: {Const.TRAP: 1, Const.CHEST: 1, Const.LOOT_PILE: 2, Const.MONSTER_SPAWNER: 1},
    Theme.DE_HEALTHY:  {Const.HEALING_STATION: 1},
    Theme.DE_GUARDED:  {Const.MONSTER_SPAWNER: 1},

    Theme.SR_TRAPPED:  {Const.HOLE: 1, Const.TRAP: (3,5), Const.LOOT_PILE: 1, Const.MONSTER_SPAWNER: 1},
    Theme.SR_TREASURE: {Const.TRAP: (1,2), Const.CHEST: 2, Const.LOOT_PILE: 3},
    Theme.SR_GUARDED:  {Const.WATER: (0,1), Const.TRAP: 1, Const.LOOT_PILE: 1, Const.MONSTER_SPAWNER: 2},
    Theme.SR_CHAOS:    {Const.HOLE: 2, Const.WATER: (0,1), Const.TRAP: 3, Const.CHEST: 1, Const.LOOT_PILE: 2, Const.MONSTER_SPAWNER: 3, Const.SHRINE: 1},
    Theme.SR_BASIC:    {Const.TRAP: (0,1), Const.LOOT_PILE: (0,1)},
    Theme.SR_FLOODED:  {Const.WATER: 5, Const.WATER_POOL: 13, Const.MONSTER_SPAWNER: 1},
    
    Theme.CN_TRAPPED:  {Const.HOLE: 1, Const.TRAP: (1,3), Const.LOOT_PILE: 1},
    Theme.CN_GUARDED:  {Const.MONSTER_SPAWNER: 1},
    Theme.CN_BASIC:    {Const.LOOT_PILE: (0,1)},
    Theme.CN_FLOODED:  {Const.WATER: 4, Const.WATER_POOL: 10, Const.MONSTER_SPAWNER: (0,1)},
    
    Theme.LR_TRAPPED:  {Const.HOLE: 2, Const.WATER: 1, Const.TRAP: (3,5), Const.LOOT_PILE: 2, Const.MONSTER_SPAWNER: 1},
    Theme.LR_TREASURE: {Const.TRAP: 1, Const.CHEST: 2, Const.LOOT_PILE: 3, Const.MONSTER_SPAWNER: 1},
    Theme.LR_HEALTHY:  {Const.HEALING_STATION: 1},
    Theme.LR_GUARDED:  {Const.WATER: (0,1), Const.TRAP: 1, Const.CHEST: 1, Const.LOOT_PILE: 1, Const.MONSTER_SPAWNER: 3},
    Theme.LR_CHAOS:    {Const.HOLE: 2, Const.WATER: 1, Const.TRAP: 3, Const.CHEST: 2, Const.LOOT_PILE: 3, Const.MONSTER_SPAWNER: (2,4), Const.SHRINE: 1},
    Theme.LR_BASIC:    {Const.TRAP: (1,2), Const.LOOT_PILE: (0,1)},
    Theme.LR_FLOODED:  {Const.WATER: 6, Const.WATER_POOL: 18, Const.MONSTER_SPAWNER: 1},
    
    Theme.CR_TRAPPED:  {Const.HOLE: 1, Const.TRAP: (2,4), Const.LOOT_PILE: 1},
    Theme.CR_TREASURE: {Const.TRAP: 1, Const.CHEST: 1, Const.LOOT_PILE: 3, Const.MONSTER_SPAWNER: 1},
    Theme.CR_GUARDED:  {Const.WATER: (0,1), Const.TRAP: 1, Const.LOOT_PILE: 1, Const.MONSTER_SPAWNER: 2},
    Theme.CR_CHAOS:    {Const.HOLE: (0,1), Const.WATER: 1, Const.TRAP: 3, Const.CHEST: (0,2), Const.LOOT_PILE: 3, Const.MONSTER_SPAWNER: (2,3), Const.SHRINE: 1},
    Theme.CR_BASIC:    {Const.TRAP: (0,1), Const.LOOT_PILE: (0,1)},
    Theme.CR_FLOODED:  {Const.WATER: 4, Const.WATER_POOL: 12, Const.MONSTER_SPAWNER: (0,1)},
    
    Theme.HR_TRAPPED:  {Const.HOLE: 1, Const.TRAP: (2,4), Const.LOOT_PILE: 1},
    Theme.HR_TREASURE: {Const.TRAP: 1, Const.CHEST: 1, Const.LOOT_PILE: 3, Const.MONSTER_SPAWNER: 1},
    Theme.HR_GUARDED:  {Const.WATER: (0,1), Const.TRAP: 1, Const.LOOT_PILE: 1, Const.MONSTER_SPAWNER: 2},
    Theme.HR_CHAOS:    {Const.HOLE: (0,1), Const.WATER: 1, Const.TRAP: 3, Const.MONSTER_SPAWNER: (2,3), Const.SHRINE: 1, Const.CHEST: (0,2), Const.LOOT_PILE: 3},
    Theme.HR_BASIC:    {Const.TRAP: (0,1), Const.LOOT_PILE: (0,1)},
    Theme.HR_FLOODED:  {Const.WATER: 5, Const.WATER_POOL: 15, Const.MONSTER_SPAWNER: (0,1)},
    
    Theme.BR_HOARD:    {Const.CHEST: 3, Const.LOOT_PILE: 9, Const.BOSS_SPAWNER: 1, Const.SHRINE: 1},
    Theme.BR_WIZARD:   {Const.CHEST: 4, Const.LOOT_PILE: 3, Const.BOSS_SPAWNER: 1, Const.SHRINE: 1},
    Theme.BR_WEAK:     {Const.TRAP: (0,1), Const.CHEST: 1, Const.LOOT_PILE: 2, Const.MONSTER_SPAWNER: 1, Const.BOSS_SPAWNER: 1},
    Theme.BR_STRONG:   {Const.HEALING_STATION: (0,1), Const.CHEST: (1,3), Const.LOOT_PILE: 5, Const.BOSS_SPAWNER: 1, Const.SHRINE: 1},
    Theme.BR_GUARDED:  {Const.TRAP: 1, Const.CHEST: 2, Const.LOOT_PILE: 3, Const.MONSTER_SPAWNER: 2, Const.BOSS_SPAWNER: 1, Const.SHRINE: 1},
    Theme.BR_DOUBLE:   {Const.CHEST: 3, Const.LOOT_PILE: 5, Const.BOSS_SPAWNER: 2, Const.SHRINE: 1},
    
    Theme.SC_TRAPPED:  {Const.HOLE: 1, Const.TRAP: (3,5), Const.LOOT_PILE: 1, Const.MONSTER_SPAWNER: 1},
    Theme.SC_TREASURE: {Const.TRAP: (1,2), Const.CHEST: 2, Const.LOOT_PILE: 3},
    Theme.SC_GUARDED:  {Const.WATER: (0,1), Const.TRAP: 1, Const.LOOT_PILE: 1, Const.MONSTER_SPAWNER: 2},
    Theme.SC_CHAOS:    {Const.HOLE: 2, Const.WATER: (0,1), Const.TRAP: 3, Const.CHEST: 1, Const.LOOT_PILE: 2, Const.MONSTER_SPAWNER: 3, Const.SHRINE: 1},
    Theme.SC_BASIC:    {Const.TRAP: (0,1), Const.LOOT_PILE: (0,1)},
    Theme.SC_FLOODED:  {Const.WATER: 4, Const.WATER_POOL: 12, Const.MONSTER_SPAWNER: (0,1)},

    Theme.LC_TRAPPED:  {Const.HOLE: 2, Const.WATER: 1, Const.TRAP: (3,5), Const.LOOT_PILE: 2, Const.MONSTER_SPAWNER: 1},
    Theme.LC_TREASURE: {Const.TRAP: 1, Const.CHEST: 2, Const.LOOT_PILE: 3, Const.MONSTER_SPAWNER: 1},
    Theme.LC_HEALTHY:  {Const.HEALING_STATION: 1},
    Theme.LC_GUARDED:  {Const.WATER: (0,1), Const.TRAP: 1, Const.CHEST: 1, Const.LOOT_PILE: 1, Const.MONSTER_SPAWNER: 3},
    Theme.LC_CHAOS:    {Const.HOLE: 2, Const.WATER: 1, Const.TRAP: 3, Const.CHEST: 2, Const.LOOT_PILE: 3, Const.MONSTER_SPAWNER: (2,4), Const.SHRINE: 1},
    Theme.LC_BASIC:    {Const.TRAP: (1,2), Const.LOOT_PILE: (0,1)},
    Theme.LC_FLOODED:  {Const.WATER: 7, Const.WATER_POOL: 21, Const.MONSTER_SPAWNER: (1,2)},

    Theme.ENTRANCE:    {Const.ENTRANCE: 1},
    Theme.EMPTY:       {}
    }

FEATURE_ORDER = (
        Const.ENTRANCE,
        Const.WATER,
        Const.WATER_POOL,
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

SCAN_PARAMS: dict[Const, dict[str, set[Const]]] = {
    Const.ENTRANCE:         {"require": {Const.FLOOR}, "place_on": {Const.WALL}},
    Const.WATER:            {"block": {Const.CHEST, Const.LOOT_PILE, Const.HOLE}},
    Const.WATER_POOL:       {"require": {Const.WATER}, "block": {Const.CHEST, Const.LOOT_PILE, Const.HOLE}},
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

DUPLICATES: dict[Const, Const] = {
    Const.WATER_POOL: Const.WATER,
    Const.LOOT_CLUSTER: Const.LOOT_PILE
}

_ROOM_Y, _ROOM_X = np.ogrid[:Const.ROOM_SIZE, :Const.ROOM_SIZE]
SMALL_CIRCLE_MASK = (
    (_ROOM_Y - Const.HALF) ** 2 + (_ROOM_X - Const.HALF) ** 2 <= 3 ** 2
)
LARGE_CIRCLE_MASK = (
    (_ROOM_Y - Const.HALF) ** 2 + (_ROOM_X - Const.HALF) ** 2 <= 6 ** 2
)

ONE_EXIT_ROOMS = [17, 18, 20, 24]