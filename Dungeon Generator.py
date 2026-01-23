"""
Dungeon Map Generator
"""

import numpy as np
from Constants import *

def init_tilemap(size: int = 15):
    return np.full((size,size), Wall, np.uint8)

def main():
    tilemap = init_tilemap()
    #Seed and Fill Room Locations
    #Room Connector
    #Room Clearing Pass
    #Room Extension Pass
    #Tilemap Trim
    return

if __name__ == "__main__":
    main()