from random import Random

import numpy as np

import Stage_1, Stage_2, Stage_3


def generate_dungeon(seed: int | None = None):
    np_rng = np.random.default_rng(seed)
    rand_rng = Random(seed)
    dungeon_map = Stage_1.map_generator(np_rng, rand_rng)
    tilemap, theme_map = Stage_2.tilemap_builder(dungeon_map, np_rng, rand_rng)
    tilemap = Stage_3.room_populator(tilemap, theme_map, np_rng)
    return tilemap, theme_map