"""
**Constants for Generators**
"""

from enum import IntEnum

class Dungeon_Generator_Constants(IntEnum):
    """
    Constants for Dungeon_Generator file
    """
    DUNGEON_SIZE = 20
    BOX_COUNT = 3
    ERODE_COUNT = 5
    NORTH = 1
    EAST = 2
    SOUTH = 4
    WEST = 8
    TEMP = 1
    NO_ROOM = 0
    ROOM = 16

class Room_Generator_Constants(IntEnum):
    """
    Constants for Room_Generator file
    """
    ROOM_SIZE = 17
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