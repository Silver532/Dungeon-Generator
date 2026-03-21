import subprocess
import tkinter as tk
from functools import partial
from random import Random
from tkinter import BooleanVar, IntVar, StringVar
from typing import Callable, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.backend_bases import Event, MouseEvent
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.text import Text
from numpy import uint8
from numpy.typing import NDArray as array

import Stage_1, Stage_2, Stage_3
from Debug import (
    enable_timing,
    make_exit_map,
    reset_timings,
    write_timings_to_file
)
from Helpers import Theme, Const

def _init_seeds(seed_str: StringVar) -> tuple[np.random.Generator, Random]:
    seed_val = seed_str.get()
    if (seed_str != "") and (seed_val != "Seed"):
        seed_val = seed_str.get()
        seed = int(seed_val) if seed_val.isdigit() else abs(hash(seed_val))
    else:
        seed = None
    np_rng = np.random.default_rng(seed)
    rand_rng = Random(seed)
    return np_rng, rand_rng

def _run_timing(
        func: Callable[[], Any],
        time_count: IntVar,
        label: str
) -> None:
    enable_timing()
    for _ in range(time_count.get()):
        func()
    enable_timing(False)
    write_timings_to_file(time_count.get())
    reset_timings()
    print(f"Test Complete:\n  {label}: {time_count.get()} runs")
    return

def _get_dungeon_info(tilemap: array[uint8], row: int, col: int) -> str:
    value = tilemap[row,col]
    bits = value & 0b01111
    directions = ('North','East','South','West')
    dir_string = ", ".join(
        direction
        for i, direction in enumerate(directions)
        if bits & (1 << i)
    )
    return f"Position: {row, col}\nExits: {dir_string}\nTile Value: {value}"

def _run_Stage_1(timing_var: BooleanVar, time_count: IntVar, seed_var: StringVar) -> None:
    subprocess.run("cls||clear", shell = True)
    if timing_var.get():
        _run_timing(
        lambda: Stage_1.map_generator(np.random.default_rng(), Random()),
        time_count,
        "Stage 1"
        )
        return
    np_rng, rand_rng = _init_seeds(seed_var)
    tilemap = Stage_1.map_generator(np_rng, rand_rng)
    debug_map = make_exit_map(tilemap)
    info = _get_dungeon_info
    colours: dict[int, str] = {
        0: "#ffffff",
        1: "#000000",
        2: "#11b411",
        3: "#0000ff",
        4: "#ff0000",
        5: "#dbdb00",
    }
    _visualize(debug_map, colours, info_func = info, click_map = tilemap)
    return

def _get_map_info(theme_map:array[uint8], tilemap: array[uint8], row: int, col: int) -> str:
    value = tilemap[row, col]
    theme = Theme(theme_map[row, col])
    return f"Position: {row, col}\nTheme: {theme.name}\nTile: {Const(value).name}"

def _run_Stage_2(timing_var: BooleanVar, time_count: IntVar, seed_var: StringVar) -> None:
    subprocess.run("cls||clear", shell = True)
    if timing_var.get():
        _run_timing(
        lambda: Stage_2.tilemap_builder(
            Stage_1.map_generator(np.random.default_rng(), Random()),
            np.random.default_rng(),
            Random()
        ),
        time_count,
        "Stage 2"
        )
        return
    np_rng, rand_rng = _init_seeds(seed_var)
    S1_tilemap = Stage_1.map_generator(np_rng, rand_rng)
    tilemap, theme_map = Stage_2.tilemap_builder(S1_tilemap, np_rng, rand_rng)
    info = partial(_get_map_info, theme_map)
    colours: dict[int, str] = {
        int(Const.WALL): "#000000",
        int(Const.FLOOR): "#ffffff",
    }
    _visualize(tilemap, colours, info_func = info, scale = 3)
    return

def _run_Stage_3(timing_var: BooleanVar, time_count: IntVar, seed_var: StringVar) -> None:
    subprocess.run("cls||clear", shell=True)
    if timing_var.get():
        _run_timing(
            lambda: Stage_3.room_populator(
                *Stage_2.tilemap_builder(
                    Stage_1.map_generator(np.random.default_rng(), Random()),
                    np.random.default_rng(),
                    Random()
                ),
                np.random.default_rng()
            ),
            time_count,
            "Stage 3"
        )
        return
    np_rng, rand_rng = _init_seeds(seed_var)
    S1_tilemap = Stage_1.map_generator(np_rng, rand_rng)
    tilemap, theme_map = Stage_2.tilemap_builder(S1_tilemap, np_rng, rand_rng)
    tilemap = Stage_3.room_populator(tilemap, theme_map, np_rng)
    info = partial(_get_map_info, theme_map)
    colours: dict[int, str] = {
        int(Const.WALL): "#000000",
        int(Const.FLOOR): "#ffffff",
        int(Const.WATER): "#1a6fcc",
        int(Const.HOLE): "#808080",
        int(Const.HEALING_STATION): "#22aa22",
        int(Const.SHRINE): "#c0c0c0",
        int(Const.CHEST): "#8b4513",
        int(Const.LOOT_PILE): "#ffd700",
        int(Const.TRAP): "#cc0000",
        int(Const.BOSS_SPAWNER): "#ff8c00",
        int(Const.MONSTER_SPAWNER): "#ff6600",
    }
    _visualize(tilemap, colours, info_func=info, scale=3)
    return

def _prepare_colours(colour_dict: dict[int, str]) -> list[str]:
    max_val = max(colour_dict.keys())
    colours: list[str] = ["#000000"] * (max_val + 1)
    for const, colour in colour_dict.items():
        colours[const] = colour
    return colours

def _on_hover(
        event: Event,
        ax: Axes,
        fig: Figure,
        target_map: array[uint8],
        info_text: Text,
        info_func: Callable[[array[uint8], int, int], str]
) -> None:
    if not isinstance(event, MouseEvent):
        return
    in_axes = (
        event.inaxes is ax
        and event.xdata is not None
        and event.ydata is not None
    )
    if not in_axes:
        return
    assert event.xdata is not None and event.ydata is not None
    col = int(event.xdata+0.5)
    row = int(event.ydata+0.5)
    in_bounds = (
        0 <= row < target_map.shape[0]
        and 0 <= col < target_map.shape[1]
    )
    if not in_bounds:
        return
    text = info_func(target_map, row, col)
    info_text.set_text(text)
    fig.canvas.draw_idle()                                                                      #pyright: ignore[reportUnknownMemberType]
    return

def _visualize(
        tilemap: array[uint8],
        colour_dict: dict[int, str],
        info_func: Callable[..., str] | None = None,
        grid_active: bool = True,
        grid_colour: str = "black",
        scale: int = 40,
        click_map: array[uint8] | None = None
) -> None:
    colours = _prepare_colours(colour_dict)
    cmap = ListedColormap(colours)
    norm = BoundaryNorm(range(len(colours) + 1), cmap.N)
    rows, cols = tilemap.shape
    rcParams["toolbar"] = "None"

    fig_width = (cols * scale) / 120
    fig_height = (rows * scale) / 120
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=120)
    ax.imshow(tilemap, cmap=cmap, norm=norm, interpolation="nearest")                          #pyright: ignore[reportUnknownMemberType]
    if grid_active: ax.grid(which="minor", color=grid_colour, linewidth=0.2)                   #pyright: ignore[reportUnknownMemberType]
    ax.tick_params(                                                                            #pyright: ignore[reportUnknownMemberType]
        which="both", bottom=False, left=False,
        labelbottom=False, labelleft=False
    )
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)                                        #pyright: ignore[reportUnknownMemberType]
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)                                        #pyright: ignore[reportUnknownMemberType]
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title("DEBUG Window")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.margins(0)
    if info_func is not None:
        target_map = click_map if click_map is not None else tilemap
        text_height = 55
        text_frac = text_height / (rows * scale + text_height)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=text_frac)
        fig_height = (rows * scale + text_height) / 120
        fig.set_size_inches(fig_width, fig_height)
        info_text = fig.text(
            0.01, 0.002, "",
            va="bottom", ha="left",
            color="#000000",
            fontsize=8
        )
        fig.canvas.mpl_connect(
            "motion_notify_event",
            partial(
                _on_hover, ax=ax, fig=fig, target_map=target_map,
                info_text=info_text, info_func=info_func
            )
        )

    plt.show(block=False)                                                                       #pyright: ignore[reportUnknownMemberType]
    return

def _main():
    subprocess.run("cls||clear", shell = True)
    root = tk.Tk()
    root.title("Debugger")
    root.geometry("225x150")

    seed_var = tk.StringVar()
    timing_var = tk.BooleanVar()
    time_count = tk.IntVar(value = 100)

    seed_entry = tk.Entry(root, textvariable = seed_var)
    seed_entry.insert(0, "Seed")
    seed_entry.bind("<FocusIn>", lambda _: seed_entry.delete(0, "end") if seed_var.get() == "Seed" else None)
    timing_check = tk.Checkbutton(root, text = "Time Test", variable = timing_var, onvalue=True, offvalue=False)
    time_label = tk.Label(root, text = "Run Count")
    tc_100 = tk.Radiobutton(root, text = "100", variable = time_count, value = 100)
    tc_1000 = tk.Radiobutton(root, text = "1000", variable = time_count, value = 1000)
    tc_10000 = tk.Radiobutton(root, text = "10000", variable = time_count, value = 10000)
    s1_button = tk.Button(root, text="Stage 1",
                          command = partial(_run_Stage_1, timing_var, time_count, seed_var))
    s2_button = tk.Button(root, text="Stage 2",
                          command = partial(_run_Stage_2, timing_var, time_count, seed_var))
    s3_button = tk.Button(root, text="Stage 3",
                          command = partial(_run_Stage_3, timing_var, time_count, seed_var))

    seed_entry.grid(row = 0, column = 0, sticky = "nsew")
    timing_check.grid(row = 0, column = 1, sticky = "w")
    time_label.grid(row = 1, column = 1, sticky = "w")
    tc_100.grid(row = 2, column = 1, sticky = "w")
    tc_1000.grid(row = 3, column = 1, sticky = "w")
    tc_10000.grid(row = 4, column = 1, sticky = "w")
    s1_button.grid(row = 1, sticky = "w")
    s2_button.grid(row = 2, sticky = "w")
    s3_button.grid(row = 3, sticky = "w")

    root.mainloop()

if __name__ == "__main__":
    _main()