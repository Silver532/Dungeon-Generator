"""
Dungeon Map Generator
"""

import numpy as np
from numpy.typing import NDArray
from random import randint as rand
from Constants import *
from os import system as sys

sys("cls")

def init_tilemap(size: int = 15):
    return np.zeros((size,size), np.uint8)

def room_fill(tilemap):
    mid = DUNGEON_SIZE//2
    for _ in range(BOX_COUNT):
        y_s, y_e = rand(1, mid - 1), rand(mid + 2, DUNGEON_SIZE - 2)
        x_s, x_e = rand(1, mid - 1), rand(mid + 2, DUNGEON_SIZE - 2)
        tilemap[y_s:y_e, x_s:x_e] = ROOM
    return tilemap

def main():
    tilemap = init_tilemap()
    tilemap = room_fill(tilemap)
    print(tilemap)
    #Room Erosion/Filling Pass
    #Room Connector
    #Room Clearing Pass
    #Room Extension Pass
    #Tilemap Trim
    return

if __name__ == "__main__":
    main()