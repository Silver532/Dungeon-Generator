from collections import deque
from random import Random

import numpy as np
from numpy import uint8
from numpy.typing import NDArray as array

from Debug import timeit
from Helpers import S1_Const

@timeit
def _init_tilemap(height: int, width: int | None = None) -> array[uint8]:
    width = width or height
    tilemap = np.zeros((height,width), dtype = uint8)
    return tilemap

@timeit
def _room_fill(
        tilemap: array[uint8],
        np_rng: np.random.Generator
) -> array[uint8]:
    for _ in range(S1_Const.BOX_COUNT):
        y_start = np_rng.integers(1, S1_Const.MID)
        y_end = np_rng.integers(S1_Const.MID + 2, S1_Const.DUNGEON_SIZE - 1)

        room_height = y_end-y_start
        room_width = 16 - room_height

        x_start = np_rng.integers(1, S1_Const.MID)
        x_end = int(np.minimum(x_start + room_width + 5, S1_Const.DUNGEON_SIZE - 2))

        tilemap[y_start:y_end, x_start:x_end] = S1_Const.TEMP
    return tilemap

@timeit
def _fast_adj(
        tilemap: array[uint8],
        neighbor_map: array[uint8]
) -> None:
    h, w = tilemap.shape
    mask = (tilemap != 0).astype(uint8)
    neighbor_map.fill(0)
    neighbor_map[1:h-1, :] = mask[0:h-2, :] + mask[2:h, :]
    neighbor_map[:, 1:w-1] += mask[:, 0:w-2] + mask[:, 2:w]
    neighbor_map *= (tilemap == 1)
    return

@timeit
def _room_eroder(
        tilemap: array[uint8],
        np_rng: np.random.Generator
) -> array[uint8]:
    neighbor_map = np.empty_like(tilemap, dtype=uint8)

    for _ in range(S1_Const.ERODE_COUNT):
        _fast_adj(tilemap, neighbor_map)

        mask_2 = (neighbor_map == 2)
        tilemap[
            mask_2 & (np_rng.random(mask_2.shape, dtype = np.float32) < 0.5)
        ] = S1_Const.NO_ROOM

        mask_3 = (neighbor_map == 3)
        tilemap[
            mask_3 & (np_rng.random(mask_3.shape, dtype = np.float32) < 0.1)
        ] = S1_Const.NO_ROOM

    _fast_adj(tilemap, neighbor_map)
    tilemap[neighbor_map == 0] = S1_Const.NO_ROOM
    return tilemap

@timeit
def _get_possible_connections(tilemap: array[uint8]) -> array[uint8]:
    t = (tilemap != 0).astype(uint8)
    connections = (
    t[:-2, 1:-1]
    | (t[1:-1, 2:] << 1)
    | (t[2:, 1:-1] << 2)
    | (t[1:-1, :-2] << 3)
    )
    connections *= t[1:-1, 1:-1]
    return connections

@timeit
def _room_random(np_rng: np.random.Generator, count: int) -> array[uint8]:
    r = np_rng.random(count, dtype = np.float32)

    randoms = np.ones(count, dtype = uint8)
    randoms[r >= 0.55] = 2
    randoms[r >= 0.80] = 3
    return randoms

@timeit
def _room_connector(
        tilemap: array[uint8],
        np_rng: np.random.Generator,
        rand_rng: Random
) -> array[uint8]:
    connection_map = _get_possible_connections(tilemap)
    H, W = tilemap.shape
    active_count = int(np.count_nonzero(tilemap != 0))
    connection_counts = _room_random(np_rng, active_count)
    MASK_TO_INDICES = tuple(
        tuple(i for i in range(4) if mask & (1 << i))
        for mask in range(16)
    )
    DIR_BITS = (1,2,4,8)
    DY_DX = ((-1,0),(0,1),(1,0),(0,-1))
    OPP_BITS = (4,8,1,2)
    randrange = rand_rng.randrange

    connection_index = 0

    for y in range(1, H - 1):
        row = tilemap[y]
        if not row.any():
            continue
        for x in range(1, W - 1):
            if row[x] == 0:
                continue

            mask: uint8 = connection_map[y - 1, x - 1] & 0b1111
            if mask == 0:
                continue

            indices = MASK_TO_INDICES[mask]
            n = len(indices)

            connect_count = connection_counts[connection_index]
            connection_index += 1

            if connect_count >= n:
                chosen = indices
            elif connect_count == 1:
                chosen = (indices[randrange(n)],)
            else:
                chosen = tuple(rand_rng.sample(indices, connect_count))

            for i in chosen:
                row[x] |= DIR_BITS[i]
                dy, dx = DY_DX[i]
                ny = y + dy
                nx = x + dx
                tilemap[ny, nx] |= OPP_BITS[i]
    return tilemap

@timeit
def _tilemap_trim(tilemap: array[uint8]) -> array[uint8]:
    active_rows = np.any(tilemap != 0, axis=1)
    active_cols = np.any(tilemap != 0, axis=0)
    trimmed_tilemap = tilemap[np.ix_(active_rows, active_cols)]
    return trimmed_tilemap

@timeit
def _room_clear(tilemap: array[uint8]) -> array[uint8]:
    DIR_OFFSETS = ((0,-1,0),(1,0,1),(2,1,0),(3,0,-1))
    h, w = tilemap.shape
    active_tiles = {(r, c) for r, c in np.argwhere(tilemap != 0).tolist()}
    groups: list[set[tuple[int, int]]] = []
    unvisited = active_tiles.copy()
    while unvisited:
        start = next(iter(unvisited))
        group: set[tuple[int, int]] = set()
        visited: set[tuple[int, int]] = {start}
        queue = deque([start])
        while queue:
            y, x = queue.popleft()
            group.add((y, x))
            val = tilemap[y, x]
            for bit, dy, dx in DIR_OFFSETS:
                if val & (1 << bit):
                    ny, nx = y + dy, x + dx
                    in_bounds: bool = 0 <= ny < h and 0 <= nx < w
                    if in_bounds and (ny, nx) not in visited:
                        visited.add((ny, nx))
                        queue.append((ny, nx))
        groups.append(group)
        unvisited -= group
    
    if groups:
        largest = max(groups, key=len)
        remainder = active_tiles - largest
        if remainder:
            to_remove = np.array(
                list(remainder), dtype=np.int32
            ).reshape(-1, 2)
            tilemap[to_remove[:, 0], to_remove[:, 1]] = 0
    return tilemap

@timeit
def map_generator(
        np_rng: np.random.Generator,
        rand_rng: Random
) -> array[uint8]:
    tilemap = _init_tilemap(S1_Const.DUNGEON_SIZE)
    tilemap = _room_fill(tilemap, np_rng)
    tilemap = _room_eroder(tilemap, np_rng)
    tilemap <<= 4
    tilemap = _room_connector(tilemap, np_rng, rand_rng)
    tilemap = _room_clear(tilemap)
    tilemap = _tilemap_trim(tilemap)
    return tilemap