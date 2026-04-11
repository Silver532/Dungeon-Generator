from functools import wraps
from time import perf_counter_ns as clock
from typing import Callable, ParamSpec, TypeVar
import os

import numpy as np
from numpy import uint8
from numpy.typing import NDArray as array

_TIMINGS: dict[str, list[float]] = {}
_timing_enabled: bool = False
P = ParamSpec("P")
R = TypeVar("R")
_DIR = os.path.dirname(os.path.abspath(__file__))

def reset_timings() -> None:
    global _TIMINGS
    _TIMINGS.clear()
    return

def enable_timing(truth: bool = True) -> None:
    global _timing_enabled
    _timing_enabled = truth
    return

def write_timings_to_file(run_count: int, filename: str = "../../timings.txt") -> None:
    path = os.path.join(_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        for name, (total, count) in _TIMINGS.items():
            avg = total/count
            per_run_calls = count / run_count
            per_run_time = total / run_count
            f.write(
                f"{name}\n"
                f"\t{'per call':<12}{avg:>9.3f} ms\n"
                f"\t{'calls':<12}{count:>9}\n"
                f"\t{'calls/run':<12}{per_run_calls:>9.1f}\n"
                f"\t{'ms/run':<12}{per_run_time:>9.3f} ms\n"
                f"\t{'total':<12}{total:>9.3f} ms\n\n"
            )
    return

def timeit(func: Callable[P,R]) -> Callable[P,R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if not _timing_enabled:
            return func(*args, **kwargs)
        start = clock()
        result = func(*args, **kwargs)
        end = clock()
        duration = (end - start) * 1e-6
        name = func.__name__
        if name not in _TIMINGS:
            _TIMINGS[name] = [0.0, 0]
        _TIMINGS[name][0] += duration
        _TIMINGS[name][1] += 1
        return result
    return wrapper

def make_exit_map(tilemap: array[uint8]) -> array[uint8]:
    debug_map = (
        np.unpackbits(tilemap[:, :, np.newaxis], axis=-1)
        .sum(axis=-1)
        .astype(np.uint8)
    )
    return debug_map