from random import Random

import numpy as np
from numpy import uint8
from numpy.typing import NDArray as array

from Helpers import (
    Const, 
    Shape, 
    Theme, 
    InvalidRoom,
    SHAPE_TABLES, 
    THEME_TABLES, 
    SMALL_CIRCLE_MASK, 
    LARGE_CIRCLE_MASK,
    ONE_EXIT_ROOMS,
    )
from Debug import timeit

@timeit
def _get_entrance_room(dungeon_map: array[uint8], rand_rng: Random):
    mask = np.isin(dungeon_map, ONE_EXIT_ROOMS)
    coords = np.argwhere(mask)
    if coords.size == 0:
        coords = np.argwhere(dungeon_map != 0)
    r, c = coords[rand_rng.randrange(coords.shape[0])]
    return int(r), int(c)

@timeit
def _init_maps(multiplier: int, h: int,w: int) -> tuple[array[uint8], array[uint8]]:
    tilemap = np.zeros((h * multiplier, w * multiplier), dtype = uint8)
    theme_map = np.zeros((h * multiplier, w * multiplier), dtype = uint8)
    return tilemap, theme_map

@timeit
def _get_shape(room_val: uint8, rand_rng: Random) -> Shape:
    if room_val < 0b10000 or room_val > 0b11111:
        raise InvalidRoom(
            f"The get_shape function does not support room_val: {room_val}."
        )
    n = (room_val & 0b01111).bit_count()
    if n not in SHAPE_TABLES:
        raise InvalidRoom(
            f"The get_shape function does not support rooms with {n} exits."
        )
    shape_list, probs = SHAPE_TABLES[n]
    room_shape: Shape = rand_rng.choices(shape_list, probs)[0]
    return room_shape

@timeit
def _build_room(
        tilemap: array[uint8],
        room_val: uint8,
        room_shape: Shape,
        np_rng: np.random.Generator
) -> array[uint8]:
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
            s = slice(half-length, half+length+1)
            tilemap[s,s] = Const.WALL
        case Shape.BOSS_ROOM:
            tilemap[1:-1, 1:-1] = Const.FLOOR
        case Shape.SMALL_ROOM:
            tilemap[half-3:half+4, half-3:half+4] = Const.FLOOR
        case Shape.CONNECTION:
            tilemap[half-1:half+2, half-1:half+2] = Const.FLOOR
        case Shape.LARGE_ROOM:
            tilemap[half-6:half+7, half-6:half+7] = Const.FLOOR
        case Shape.CORNER:
            match int(room_val) & 0b01111:
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
            match int(room_val) & 0b01111:
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
        case Shape.SMALL_CIRCLE:
            tilemap[SMALL_CIRCLE_MASK] = Const.FLOOR
        case Shape.LARGE_CIRCLE:
            tilemap[LARGE_CIRCLE_MASK] = Const.FLOOR
    return tilemap

@timeit
def _get_theme(room_shape: Shape, rand_rng: Random) -> Theme:
    if room_shape not in THEME_TABLES:
        raise InvalidRoom(f"The get_theme function does not support rooms with {room_shape.name} shape")
    theme_list, probs = THEME_TABLES[room_shape]
    theme: Theme = rand_rng.choices(theme_list, probs)[0]
    return theme

@timeit
def tilemap_builder(
        dungeon_map: array[uint8],
        np_rng: np.random.Generator,
        rand_rng: Random
) -> tuple[array[uint8],array[uint8]]:
    multiplier = Const.ROOM_SIZE
    tilemap, theme_map = _init_maps(multiplier, *dungeon_map.shape)
    entrance = _get_entrance_room(dungeon_map, rand_rng)
    room_tilemap = np.empty(
        shape = (Const.ROOM_SIZE, Const.ROOM_SIZE),
        dtype = uint8
    )
    for row, col in np.argwhere(dungeon_map != 0):
        room_tilemap.fill(0)
        val = uint8(dungeon_map[row, col])
        if (row, col) == entrance:
            theme = Theme.ENTRANCE
            shape = Shape.DEAD_END
        else:
            shape = _get_shape(val, rand_rng)
            theme = _get_theme(shape, rand_rng)
        room_tilemap = _build_room(room_tilemap, val, shape, np_rng)
        y = row * multiplier
        x = col * multiplier
        tilemap[y:y + multiplier, x:x + multiplier] = room_tilemap
        theme_map[y:y + multiplier, x:x + multiplier] = theme
    return tilemap, theme_map