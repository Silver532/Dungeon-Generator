"""
Debug Tools for the entire program
"""

import atexit
import functools

from time import perf_counter_ns as clock
from typing import Callable, Any

_TIMINGS: dict[str, list[float]] = {}
_timing_enabled: bool = False

def enable_timing() -> None:
    global _timing_enabled
    _timing_enabled = True

def write_timings_to_file(filename: str = "timings.txt") -> None:
    with open(filename, "w", encoding="utf-8") as f:
        for name, (total, count) in _TIMINGS.items():
            avg = total/count
            f.write(
                f"{name:<21}  "
                f"avg={avg:>6.3f} ms  "
                f"calls={count:>6}  "
                f"total={total:>8.3f} ms\n"
            )
    return

def timeit(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
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

atexit.register(write_timings_to_file)