import numpy as np
from numpy import uint8
from numpy.typing import NDArray as array

from Constants import *
from Generator_Helpers import *

def room_map_generator() -> array[uint8]:
    tilemap = init_tilemap(25)
    return tilemap

def _main() -> None:
    from time import perf_counter_ns as clock
    print("\033c", end="")
    start_time = clock()

    tilemap = room_map_generator()

    end_time = clock()
    delta_time = (end_time - start_time)/1000000
    print(f"Program ran in {delta_time} milliseconds")
    return

if __name__ == "__main__":
    _main()