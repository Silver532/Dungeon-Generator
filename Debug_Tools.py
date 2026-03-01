"""
Debug Tools for the entire program
"""

import atexit
import functools
import argparse

from time import perf_counter_ns as clock
from typing import Callable, TypeVar, ParamSpec

_TIMINGS: dict[str, list[float]] = {}
_timing_enabled: bool = False
P = ParamSpec("P")
R = TypeVar("R")

def enable_timing() -> None:
    """
    Enable performance timing globally for all decorated functions.
    """
    global _timing_enabled
    _timing_enabled = True

def arg_parser() -> bool:
    """
    Parses command-line arguments to check for timing flag.

    Returns
    -------
    bool
        True if --time flag is provided, otherwise False.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", action = "store_true", help = "Enable performance timing")
    args = parser.parse_args()
    if args.time: enable_timing(); return True
    return False

def write_timings_to_file(filename: str = "timings.txt") -> None:
    """
    Writes collected timing statistics to a file.

    Parameters
    ----------
    filename : str = timings.txt
        Output file name.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for name, (total, count) in _TIMINGS.items():
            avg = total/count
            f.write(
                f"{name:<25}  "
                f"avg={avg:>7.3f} ms  "
                f"calls={count:>6}  "
                f"total={total:>9.3f} ms\n"
            )
    return

def timeit(func: Callable[P,R]) -> Callable[P,R]:
    """
    Decorator that measures execution time of a function.

    Parameters
    ----------
    func : Callable
        Function to be wrapped and timed.

    Returns
    -------
    Callable
        Wrapped function that records execution time when timing is enabled.
    """
    @functools.wraps(func)
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

atexit.register(write_timings_to_file)