#!/usr/bin/env python3
"""Editable reproduction of Fig. 1; run `python3 plot_fig1.py`.

Each panel has its own coordinates, with the origin at its upper-left corner.
Coordinates follow the reference at a convenient 1824 x 1368 design resolution;
they are independent of export resolution. Change PANEL_BOUNDS to move panels.
All outputs and Matplotlib's cache are kept beside this script.
"""

from pathlib import Path
import os

HERE = Path(__file__).resolve().parent
os.environ["MPLCONFIGDIR"] = str(HERE / ".matplotlib")
os.environ["TMPDIR"] = str(HERE)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
import numpy as np

# Register Arial explicitly on macOS, while allowing normal font discovery on
# other systems. Fail clearly rather than silently substitute another family.
for filename in ("Arial.ttf", "Arial Bold.ttf", "Arial Italic.ttf"):
    font_path = Path("/System/Library/Fonts/Supplemental") / filename
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
font_manager.findfont("Arial", fallback_to_default=False)

matplotlib.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.6,
    "savefig.facecolor": "white",
})

FIGURE_SIZE = (7.2, 5.4)
DESIGN_SIZE = (1824, 1368)
# Node colours (Okabe-Ito blue, orange and green; grey for the redraw in panel a).
SHADES = {"orange": "#E69F00", "green": "#009E73", "blue": "#0072B2", "grey": "#555555"}
PANEL_BOUNDS = {
    "a": (87, 38, 810, 830),
    "b": (957, 38, 783, 368),
    "c": (957, 432, 783, 440),
    "d": (87, 899, 1664, 446),
}


def node(ax, x, y, shade, radius=24):
    """Draw one node above its incident edges; radius is in panel units."""
    artist = Circle((x, y), radius, facecolor=SHADES[shade],
                    edgecolor="none", zorder=3)
    ax.add_patch(artist)
    return artist


def edge(ax, start, end, width=1.9, dotted=False):
    """Draw an edge or a closely dotted stub; width is in points."""
    line, = ax.plot([start[0], end[0]], [start[1], end[1]],
                    color="black", linewidth=width,
                    linestyle=(0, (1, 1)) if dotted else "-",
                    solid_capstyle="butt", dash_capstyle="butt", zorder=1)
    return line


def arrow(ax, start, end, *, double=False, dotted=False, width=1.5,
          head_length=24, head_width=24):
    """Draw a shaft and filled triangular heads with predictable geometry.

    Start/end are the visible arrow tips, not the centres of connected nodes.
    Arrowhead dimensions use the same local units as nodes and positions.
    """
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    direction = (end - start) / np.linalg.norm(end - start)
    normal = np.array([-direction[1], direction[0]])
    end_base = end - direction * head_length
    start_base = start + direction * head_length if double else start
    edge(ax, start_base, end_base, width=width, dotted=dotted)
    ax.add_patch(Polygon([end, end_base + normal * head_width / 2,
                          end_base - normal * head_width / 2],
                         closed=True, facecolor="black", edgecolor="none",
                         zorder=2))
    if double:
        ax.add_patch(Polygon([start, start_base + normal * head_width / 2,
                              start_base - normal * head_width / 2],
                             closed=True, facecolor="black", edgecolor="none",
                             zorder=2))


def frame(ax, x, y, width, height, *, radius=48, linewidth=0.65):
    """Rounded outline, including the heavier neighbour-colour boxes."""
    artist = FancyBboxPatch((x, y), width, height,
                            boxstyle=f"round,pad=0,rounding_size={radius}",
                            facecolor="none", edgecolor="black",
                            linewidth=linewidth, clip_on=False, zorder=0)
    ax.add_patch(artist)
    return artist


def label(ax, x, y, text, *, size=9, bold=False, ha="center"):
    """Text is centred vertically; all ordinary text remains editable."""
    return ax.text(x, y, text, fontsize=size,
                   fontweight="bold" if bold else "normal",
                   ha=ha, va="center", linespacing=1.12, zorder=4)


def panel(fig, name):
    """Create an independent panel coordinate system with y increasing down."""
    x, y, width, height = PANEL_BOUNDS[name]
    dw, dh = DESIGN_SIZE
    ax = fig.add_axes([x / dw, 1 - (y + height) / dh,
                       width / dw, height / dh])
    ax.set(xlim=(0, width), ylim=(height, 0), aspect="equal")
    ax.set_axis_off()
    frame(ax, 0, 0, width, height)
    label(ax, 46, 64 if name != "d" else 59, name,
          size=11, bold=True, ha="left")
    return ax


def draw_a(fig):
    ax = panel(fig, "a")
    # A tree whose two adjacent dark central nodes are in conflict.
    positions = {
        "upper_mid": (457, 64, "green"),
        "upper_dark": (516, 209, "blue"),
        "upper_light": (731, 175, "orange"),
        "lower_dark": (253, 374, "blue"),
        "lower_light": (121, 475, "orange"),
        "lower_mid": (335, 537, "green"),
    }
    for source, target in [("upper_mid", "upper_dark"),
                           ("upper_light", "upper_dark"),
                           ("upper_dark", "lower_dark"),
                           ("lower_dark", "lower_light"),
                           ("lower_dark", "lower_mid")]:
        edge(ax, positions[source][:2], positions[target][:2])
    for x, y, shade in positions.values():
        node(ax, x, y, shade)
    label(ax, 269, 219, "Gridlock", size=9.5)
    node(ax, 180, 680, "blue")
    node(ax, 652, 680, "grey")
    arrow(ax, (270, 680), (561, 680), double=True,
          width=2, head_length=30, head_width=30)
    label(ax, 180, 755, "Stay", size=9.5)
    label(ax, 651, 755, "Random choice", size=9.5)


def draw_b(fig):
    ax = panel(fig, "b")
    label(ax, 431, 57, "Synchronous Noise Model (SNM)")
    for y in (140, 296):
        edge(ax, (160, y + 3), (643, y + 3), dotted=True, width=0.9)
        edge(ax, (270, y + 3), (540, y + 3))
        node(ax, 270, y, "blue", radius=30)
        node(ax, 540, y, "blue", radius=30)
    arrow(ax, (268, 181), (268, 261))
    arrow(ax, (538, 181), (538, 261))
    arrow(ax, (559, 181), (612, 263), dotted=True)
    node(ax, 630, 298, "orange", radius=30)
    label(ax, 679, 210, r"Noise $\varepsilon$")


def draw_c(fig):
    ax = panel(fig, "c")
    label(ax, 458, 57, "Asynchronous Random Model (ARM)")
    edge(ax, (157, 134), (641, 134), dotted=True, width=0.9)
    edge(ax, (266, 135), (537, 135))
    node(ax, 266, 132, "blue", radius=30)
    node(ax, 537, 132, "blue", radius=30)
    arrow(ax, (264, 178), (264, 243))
    label(ax, 162, 206, "Random")
    edge(ax, (157, 285), (371, 285), dotted=True, width=0.9)
    node(ax, 266, 283, "orange", radius=30)
    arrow(ax, (537, 180), (537, 334))
    label(ax, 642, 249, r"Freezing $p$")
    edge(ax, (427, 391), (643, 391), dotted=True, width=0.9)
    node(ax, 537, 389, "blue", radius=30)
    arrow(ax, (313, 313), (485, 370), double=True)


def draw_d(fig):
    ax = panel(fig, "d")
    # Three decision sub-panels, with the rightmost split into SNM and ARM.
    edge(ax, (543, 70), (543, 410), width=0.9)
    edge(ax, (1020, 70), (1020, 410), width=0.9)

    # Left: use the neighbours' colours to make a conflict-free move.
    frame(ax, 194, 39, 225, 111, radius=29, linewidth=1.55)
    node(ax, 262, 95, "orange")
    node(ax, 353, 95, "green")
    label(ax, 306, 191, "Neighbors' colors")
    node(ax, 198, 300, "orange")
    node(ax, 416, 300, "blue")
    arrow(ax, (261, 300), (351, 300))
    label(ax, 306, 368, "Move to the conflict-free color", size=8.5)

    # Middle: either remaining colour resolves the conflict.
    frame(ax, 677, 39, 225, 111, radius=29, linewidth=1.55)
    node(ax, 790, 95, "orange")
    node(ax, 673, 296, "orange")
    node(ax, 850, 250, "blue")
    node(ax, 850, 342, "green")
    arrow(ax, (707, 296), (798, 259))
    arrow(ax, (707, 296), (798, 333))
    label(ax, 782, 407, "Choose among\nconflict-free colors", size=8.5)

    # Right: all three colours are occupied, so noise/randomness is needed.
    label(ax, 1130, 95, "Gridlock")
    frame(ax, 1220, 39, 225, 111, radius=29, linewidth=1.55)
    node(ax, 1263, 95, "orange")
    node(ax, 1333, 95, "green")
    node(ax, 1407, 95, "blue")
    label(ax, 1580, 101, "Random\nchoice")

    label(ax, 1168, 245, "Stay")
    node(ax, 1087, 297, "orange")
    node(ax, 1254, 297, "orange")
    arrow(ax, (1125, 297), (1215, 297))
    label(ax, 1168, 393, "SNM")
    edge(ax, (1342, 199), (1342, 385), dotted=True, width=0.9)

    node(ax, 1420, 300, "orange")
    node(ax, 1587, 190, "blue")
    node(ax, 1587, 298, "orange")
    node(ax, 1587, 398, "green")
    for end in [(1539, 215), (1548, 298), (1537, 380)]:
        arrow(ax, (1456, 298), end)
    label(ax, 1457, 398, "ARM")


def validate_text(fig):
    """Check rendered text bounds against panel and decision-column bounds."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for text in ax.texts:
            bounds = text.get_window_extent(renderer)
            if not ax.bbox.contains(bounds.x0, bounds.y0) or not ax.bbox.contains(
                    bounds.x1, bounds.y1):
                raise RuntimeError(f"Label extends outside panel: {text.get_text()}")
    ax = fig.axes[3]
    column_bounds = {
        "Neighbors' colors": (0, 543),
        "Move to the conflict-free color": (0, 543),
        "Choose among\nconflict-free colors": (543, 1020),
    }
    for text in ax.texts:
        if text.get_text() in column_bounds:
            bounds = text.get_window_extent(renderer).transformed(
                ax.transData.inverted())
            left, right = column_bounds[text.get_text()]
            if bounds.x0 < left + 15 or bounds.x1 > right - 15:
                raise RuntimeError(f"Label exceeds its decision column: {text.get_text()}")


def main():
    fig = plt.figure(figsize=FIGURE_SIZE, dpi=300, facecolor="white")
    draw_a(fig)
    draw_b(fig)
    draw_c(fig)
    draw_d(fig)
    validate_text(fig)
    # Do not use bbox_inches='tight': preserve the exact 7.2 x 5.4 in canvas.
    for extension in ("pdf", "svg", "png"):
        destination = HERE / f"fig1.{extension}"
        fig.savefig(destination, dpi=300)
        print(f"Wrote {destination.name}")
    plt.close(fig)


if __name__ == "__main__":
    main()
