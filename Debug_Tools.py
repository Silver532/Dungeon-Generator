"""
**Debug Tools for the entire program**
"""

from atexit import register
from functools import wraps
from argparse import ArgumentParser

from time import perf_counter_ns as clock
from typing import Callable, TypeVar, ParamSpec

_TIMINGS: dict[str, list[float]] = {}
_timing_enabled: bool = False
P = ParamSpec("P")
R = TypeVar("R")

def enable_timing() -> None:
    """
    Enables performance timing globally for all functions decorated
    with @timeit.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    - Sets the global flag _timing_enabled to True.
    - Once enabled, all @timeit decorated functions will measure and
      report their execution time on each call.
    - Timing is disabled by default and must be explicitly enabled
      by calling this function.
    """
    global _timing_enabled
    _timing_enabled = True

def arg_parser() -> bool:
    """
    Parses command-line arguments to check for the timing flag.

    Parameters
    ----------
    None

    Returns
    -------
    bool
        True if the --time flag is provided, otherwise False.

    Notes
    -----
    - Uses ArgumentParser to handle command-line arguments.
    - The only supported flag is --time, which is a boolean store_true
      argument (no value required, presence alone sets it to True).
    - If --time is provided, enable_timing() is called to activate
      performance timing globally before returning True.
    - If --time is not provided, returns False without any side effects.
    """
    parser = ArgumentParser()
    parser.add_argument("--time", action = "store_true", help = "Enable performance timing")
    args = parser.parse_args()
    if args.time: enable_timing(); return True
    return False

def write_timings_to_file(filename: str = "timings.txt") -> None:
    """
    Writes collected timing statistics from all @timeit decorated
    functions to a file.

    Parameters
    ----------
    filename : str, optional
        Output file name. Defaults to 'timings.txt'.

    Returns
    -------
    None

    Notes
    -----
    - Reads timing data from the global _TIMINGS dictionary, which is
      populated by the @timeit decorator during execution.
    - For each recorded function, the following statistics are written
      as a single formatted line:
        - Function name
        - Average time    : Time per call in milliseconds
        - Call count      : Total number of times the function was called
        - Total time      : Cumulative time across all calls in milliseconds
    - The file is written in UTF-8 encoding, overwriting any existing
      file with the same name.
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
    Decorator that measures and records the execution time of a function.

    Parameters
    ----------
    func : Callable
        Function to be wrapped and timed.

    Returns
    -------
    Callable
        Wrapped function that records execution time when timing is enabled.

    Notes
    -----
    - If timing is disabled (i.e. _timing_enabled is False), the wrapped
      function behaves identically to the original with no overhead.
    - When timing is enabled:
        - The function is called between two clock() readings.
        - The elapsed time is converted from nanoseconds to milliseconds
          via multiplication by 1e-6.
        - Results are stored in the global _TIMINGS dictionary under the
          function's name, accumulating total time and call count for use
          by write_timings_to_file().
    - Uses @wraps to preserve the original function's name, docstring,
      and other metadata on the wrapper.
    - Uses ParamSpec (P) and TypeVar (R) to ensure the wrapper's type
      signature exactly matches the wrapped function's, maintaining full
      type safety.
    """
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

register(write_timings_to_file)