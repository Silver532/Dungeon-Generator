"""
**Debug Tools for the entire program**

Imported by: Dungeon_Generator, Room_Generator, Tilemap_Stitcher
"""

from argparse import ArgumentParser
from atexit import register
from collections.abc import Mapping
from functools import wraps
from time import perf_counter_ns as clock
from typing import Callable, ParamSpec, TypeVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event, MouseEvent
from matplotlib.colors import BoundaryNorm, ListedColormap
from numpy import uint8
from numpy.typing import NDArray as array

_TIMINGS: dict[str, list[float]] = {}
_RUN_KEYS = ("dungeon_generator", "dungeon_map_generator", "room_map_generator")
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
    parser.add_argument(
        "--time",
        action = "store_true",
        help = "Enable performance timing"
    )
    args = parser.parse_args()
    if args.time: enable_timing(); return True
    return False

def _get_run_count() -> float:
    for key in _RUN_KEYS:
        if key in _TIMINGS:
            return _TIMINGS[key][1]
    return 1

def _write_timings_to_file(filename: str = "../timings.txt") -> None:
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
    run_count = _get_run_count()
    with open(filename, "w", encoding="utf-8") as f:
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

def on_click(
        event: Event,
        ax: Axes,
        tilemap: array[uint8],
        info: Mapping[str, str | int],
        tile_formatter: Callable[[int], tuple[str,str]] | None = None
) -> None:
    """
    Handles debug click events on a matplotlib figure.

    Parameters
    ----------
    event : Event
        Matplotlib click event.
    ax : Axes
        Matplotlib graph axes, used to determine whether the click
        landed inside or outside the display area.
    tilemap : array[uint8]
        Tilemap to read tile values from on click. May differ from the
        display tilemap if click_map was provided to debug_render.
    info : Mapping[str, str | int]
        Key-value pairs to print on every click, such as room shape,
        theme, or dungeon statistics.
    tile_formatter : Callable[[int], tuple[str, str]] | None, optional
        If provided, called with the raw tile value to produce a
        formatted label and value string for display. If not provided,
        the raw integer value is printed instead.

    Returns
    -------
    None

    Notes
    -----
    - Only responds to MouseEvent types; all other event types are ignored.
    - The console is cleared before each print using the '\\033c' escape code.
    - If the click lands inside the axes on a valid tile:
        - The clicked pixel coordinates are rounded to the nearest tile index.
        - The info dict, tile position, and raw tile value are printed.
        - If tile_formatter is provided, its output is printed as an
          additional labelled line below the raw value.
    - If the click lands outside the axes, only the info dict is printed.
    """
    if not isinstance(event, MouseEvent):
        return
    info_str = "\n".join(f"{k}: {v}" for k, v in info.items())
    in_axes = (
        event.inaxes is ax
        and event.xdata is not None
        and event.ydata is not None
    )
    if in_axes:
        assert event.xdata is not None and event.ydata is not None
        col = int(event.xdata+0.5)
        row = int(event.ydata+0.5)
        in_bounds = (
            0 <= row < tilemap.shape[0]
            and 0 <= col < tilemap.shape[1]
        )
        if in_bounds:
            val = int(tilemap[row, col])
            tile_info = f"\nTile Clicked: {row}, {col}"\
                        f"\nTile Value: {val}"
            if tile_formatter is not None:
                formatter_name, tile_str = tile_formatter(val)
                print(f"\033c{info_str}{tile_info}\n"
                      f"{formatter_name}: {tile_str}"
                )
            else:
                print(f"\033c{info_str}{tile_info}")
    else:
        print(f"\033c{info_str}")
    return

def debug_render(
        tilemap: array[uint8],
        colours: list[str],
        info: Mapping[str,str | int] | None = None, 
        grid_colour: str | None = "black",
        figsize: tuple[float, float] = (5,5),
        tile_formatter: Callable[[int], tuple[str,str]] | None = None,
        click_map: array[uint8] | None = None
) -> None:
    """
    Renders an interactive debug visualization of a tilemap.

    Parameters
    ----------
    tilemap : array[uint8]
        Tilemap to display.
    colours : list[str]
        Colour palette mapping tile values to display colours. The list
        index corresponds directly to the tile value.
    info : Mapping[str, str | int] | None, optional
        Key-value pairs to print on click. If not provided, no click
        handler is attached.
    grid_colour : str, optional
        Colour of the grid lines drawn between tiles. Defaults to "black".
    figsize : tuple[float, float], optional
        Figure size in inches as (width, height). Defaults to (5, 5).
    tile_formatter : Callable[[int], tuple[str, str]] | None, optional
        If provided, passed to on_click to format raw tile values into
        a labelled string for display.
    click_map : array[uint8] | None, optional
        If provided, tile values are read from this array on click instead
        of tilemap. Useful when the display tilemap has been transformed
        and no longer holds the original tile values.

    Returns
    -------
    None

    Notes
    -----
    - The matplotlib toolbar is hidden via rcParams for a cleaner window.
    - The figure is rendered at 120 DPI.
    - Minor ticks are placed at half-integer positions to draw grid lines
      between tiles rather than through them.
    - All tick marks and axis labels are hidden, leaving only the colour
      grid visible.
    - The window title is set to "DEBUG Window" if the canvas manager
      supports it.
    - If info is provided, a click event listener is connected to the
      figure, delegating all handling to on_click().
    - Several matplotlib calls are marked with pyright: ignore
      [reportUnknownMemberType] due to false positives from incomplete
      matplotlib type stubs.
    """
    cmap = ListedColormap(colours)
    norm = BoundaryNorm(range(len(colours)+1), cmap.N)
    rows, cols = tilemap.shape
    rcParams["toolbar"]="None"
    
    fig, ax = plt.subplots(figsize = figsize, dpi = 120)                                        #pyright: ignore[reportUnknownMemberType]
    ax.imshow(tilemap,cmap=cmap,norm=norm,interpolation="nearest")                              #pyright: ignore[reportUnknownMemberType]
    if grid_colour is not None: ax.grid(which="minor", color=grid_colour, linewidth=0.2)        #pyright: ignore[reportUnknownMemberType]
    ax.tick_params(                                                                             #pyright: ignore[reportUnknownMemberType]
        which="both", bottom=False, left=False,
        labelbottom=False, labelleft=False
    )
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)                                         #pyright: ignore[reportUnknownMemberType]
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)                                         #pyright: ignore[reportUnknownMemberType]

    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title("DEBUG Window")
    if info is not None:
        target_map = click_map if click_map is not None else tilemap
        fig.canvas.mpl_connect(
            "button_press_event",
            lambda event: on_click(event, ax, target_map,info, tile_formatter)
        )
    plt.show()                                                                                  #pyright: ignore[reportUnknownMemberType]
    return

register(lambda: _write_timings_to_file() if _TIMINGS else None)