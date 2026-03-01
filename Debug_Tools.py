"""
Debug Tools for the entire program
"""

import atexit
import functools

from time import perf_counter_ns as clock
from typing import Callable, Any

_TIMINGS: list[tuple[str, float]] = []

def write_timings_to_file(filename: str = "timings.txt") -> None:
    with open(filename, "w", encoding="utf-8") as f:
        for name, duration in _TIMINGS:
            f.write(f"{name} executed in {duration} ms")
    return

def timeit(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = clock()
        result = func(*args, **kwargs)
        end = clock()
        duration = (end - start) * 1e-6
        _TIMINGS.append((func.__name__, duration))
        return result
    return wrapper

atexit.register(write_timings_to_file)