from collections.abc import Collection

import numpy as np
from numpy import uint8
from numpy.typing import NDArray as array

from Debug import timeit
from Helpers import (
    Const,
    Theme,
    POPULATION_TABLES,
    FEATURE_ORDER,
    SCAN_PARAMS,
    DUPLICATES,
)

@timeit
def _resolve_counts(
        theme_map: array[uint8],
        np_rng: np.random.Generator
) -> dict[tuple[int,int],dict[Const,int]]:
    counts: dict[tuple[int, int], dict[Const, int]] = {}
    room_themes = theme_map[::Const.ROOM_SIZE, ::Const.ROOM_SIZE]
    for row, col in np.argwhere(room_themes != 0):
        theme = Theme(int(room_themes[row, col]))
        resolved: dict[Const, int] = {}
        for feature, count in POPULATION_TABLES[theme].items():
            if isinstance(count, tuple):
                resolved[feature] = int(np_rng.integers(
                    count[0], count[1], endpoint=True
                ))
            else:
                resolved[feature] = count
        counts[int(row), int(col)] = resolved
    return counts

@timeit
def _adj_map(
            tilemap: array[uint8],
            neighbor_map:array[uint8] | None = None,
            iso:bool=True,
            target:Collection[int] | None = None
) -> array[uint8]:
    h, w = tilemap.shape
    if target is not None:
        if len(target) == 1:
            mask = (tilemap == next(iter(target))).astype(uint8)
        else:
            mask = np.isin(tilemap, tuple(target)).astype(uint8)
    else:
        mask = (tilemap != 0).astype(uint8)
    if neighbor_map is None:
        neighbor_map = np.zeros_like(tilemap, dtype = uint8)
    else: neighbor_map.fill(0)

    neighbor_map[1:h-1, :] = mask[0:h-2, :] + mask[2:h, :]
    neighbor_map[:, 1:w-1] += mask[:, 0:w-2] + mask[:, 2:w]
    if iso: neighbor_map *= (tilemap == 1)
    assert neighbor_map is not None
    return neighbor_map

@timeit
def _scan_tilemap(
        tilemap: array[uint8],
        neighbor_map: array[uint8] | None = None,
        require: Collection[int] | None = None,
        block: Collection[int] | None = None,
        bias: Collection[int] | None = None,
        place_on: Collection[int] | None = None
) -> array[np.int32]:
    if place_on is None: available_grid = tilemap == Const.FLOOR
    else: available_grid = np.isin(tilemap,tuple(place_on))

    if neighbor_map is None:
        neighbor_map = np.empty_like(tilemap, dtype=uint8)

    if require is not None:
        available_grid &= (
            _adj_map(tilemap, neighbor_map, target = require, iso = False) != 0
        )
    if block is not None:
        available_grid &= (
            _adj_map(tilemap, neighbor_map, target = block, iso = False) == 0
        )

    available_list = np.argwhere(available_grid).astype(np.int32, copy=False)
    if bias is not None:
        bias_grid = (
            _adj_map(tilemap, neighbor_map, target=bias, iso=False) != 0
        )
        bias_mask = bias_grid[available_list[:, 0], available_list[:, 1]]
        biases = available_list[bias_mask]
        if biases.size > 0:
            bias_list = np.repeat(biases, 4, axis=0)
            available_list = np.concatenate(
                (available_list,bias_list),axis = 0
            )
    return available_list

@timeit
def _place(
        tilemap: array[uint8],
        feature: Const,
        available_list: array[np.int32],
        count: int,
        np_rng: np.random.Generator) -> None:
    count = min(count, len(available_list))
    if not count: return
    indices = np_rng.choice(len(available_list), size=count, replace=False)
    coords = available_list[indices]
    if feature in DUPLICATES:
        tile = DUPLICATES[feature]
    else:
        tile = feature
    tilemap[coords[:, 0], coords[:, 1]] = tile
    return

@timeit
def room_populator(
        tilemap: array[uint8],
        theme_map: array[uint8],
        np_rng: np.random.Generator
) -> array[uint8]:
    feature_mapping = _resolve_counts(theme_map, np_rng)
    neighbor_map = np.empty_like(tilemap, dtype=uint8)
    rs = Const.ROOM_SIZE

    for (room_row, room_col), resolved in feature_mapping.items():
        y = room_row * rs
        x = room_col * rs
        room_view = tilemap[y:y + rs, x:x + rs]
        room_neighbor = neighbor_map[y:y + rs, x:x + rs]

        for feature in FEATURE_ORDER:
            count = resolved.get(feature, 0)
            if not count:
                continue
            available_list = _scan_tilemap(
                room_view, room_neighbor, **SCAN_PARAMS[feature]
            )
            if available_list.size == 0:
                continue
            _place(room_view, feature, available_list, count, np_rng)

    return tilemap