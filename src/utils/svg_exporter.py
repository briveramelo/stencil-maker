"""
Utility to convert boolean masks into minimal SVG stencil layers.

Features
--------
* One <path> per colour layer (compound path with holes via even‑odd fill rule).
* 4‑connected edge tracing → axis‑aligned, pixel‑perfect outlines (no diagonals).
* No extra dependencies beyond NumPy, OpenCV, and svgwrite.
"""

from pathlib import Path
from typing import Iterable
import numpy as np
import svgwrite

from src.models.models import RGB, RGBA, Point, Loop


def masks_to_svgs(
    masks: Iterable[np.ndarray],
    rgba_palette: list[RGBA],
    out_dir: Path,
    scale: int,
    base_filename: str,
) -> None:
    """
    Export each mask as an SVG file.

    Parameters
    ----------
    masks
        Iterable of H×W numpy arrays with truthy pixels marking filled areas.
    rgba_palette
        Parallel list of RGBA tuples; only RGB is used to choose filename/fill.
    out_dir
        Folder where outputs are written. Created if it does not yet exist.
    scale
        Document pixel size in the SVG (does not change path geometry).
    base_filename
        Output prefix — e.g. base_white.svg, base_color1.svg, …
    """
    masks = list(masks)  # may be a generator
    out_dir.mkdir(parents=True, exist_ok=True)

    height, width = masks[0].shape
    doc_px_size = (width * scale, height * scale)

    # Index counter used to assign sequential names (color1, color2, …) to non‑white/black layers
    special_color_index = 1
    for mask, rgba in zip(masks, rgba_palette):
        rgb: RGB = rgba[:3]
        color_label, special_color_index = _get_color_name(rgb, special_color_index)

        dwg = svgwrite.Drawing(
            filename=str(out_dir / f"{base_filename}_{color_label}.svg"),
            size=doc_px_size,
            viewBox=f"0 0 {width} {height}",
        )

        path_d = _trace_contours_as_svg_paths(mask)
        if path_d:
            dwg.add(
                dwg.path(
                    d=path_d,
                    fill=f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}",
                    fill_rule="evenodd",
                )
            )
        dwg.save()


def _get_color_name(rgb: RGB, other_index: int) -> tuple[str, int]:
    """Return a predictable colour label for filenames."""
    tol = 10
    if all(abs(c - 255) <= tol for c in rgb):
        return "white", other_index
    if all(c <= tol for c in rgb):
        return "black", other_index
    return f"color{other_index}", other_index + 1


def _trace_contours_as_svg_paths(mask: np.ndarray) -> str:
    """
    Convert a binary mask into a single SVG path string (compound with holes).

    Uses a custom 4‑connected border trace so that all segments are axis‑aligned
    and lie exactly on pixel boundaries — perfect for cutters.
    """
    loops = _trace_edges(mask)
    if not loops:
        return ""

    sub_paths = []
    for loop in loops:
        # _add_bridges(loop, 2, .5)
        pieces = [f"M {loop[0][0]} {loop[0][1]}"] # Begin the sub‑path with a Move‑to (M) command at the first vertex
        pieces.extend(f"L {x} {y}" for x, y in loop[1:]) # Append Line‑to (L) commands for every remaining vertex
        pieces.append("Z") # Close the current sub‑path (Z) to finish the loop
        sub_paths.append(" ".join(pieces))

    return " ".join(sub_paths)


def _trace_edges(mask: np.ndarray) -> list[Loop]:
    """
    Trace 4‑connected pixel edges and return ordered loops.

    Returns a list of loops; each loop is a list of integer (x, y) vertices on
    pixel boundaries.  Even‑odd fill handles holes automatically.
    """
    mask = mask.astype(bool)
    h, w = mask.shape
    # represents every exposed border segment of the binary mask as an undirected graph
    adj: dict[
        Point, # (x, y) locations where horizontal and vertical edges meet
        set[Point] # set of other (x, y) vertices that share a border with the key
    ] = {}

    def _add_edge(v1: Point, v2: Point) -> None:
        adj.setdefault(v1, set()).add(v2)
        adj.setdefault(v2, set()).add(v1)

    # Build undirected graph of exposed edges
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            # Top edge
            if y == 0 or not mask[y - 1, x]:
                _add_edge((x, y), (x + 1, y))
            # Bottom edge
            if y == h - 1 or not mask[y + 1, x]:
                _add_edge((x + 1, y + 1), (x, y + 1))
            # Left edge
            if x == 0 or not mask[y, x - 1]:
                _add_edge((x, y + 1), (x, y))
            # Right edge
            if x == w - 1 or not mask[y, x + 1]:
                _add_edge((x + 1, y), (x + 1, y + 1))

    loops: list[Loop] = []

    def _pop_edge(v1: Point, v2: Point) -> None:
        """Remove the undirected edge from both vertices; delete the key if its set becomes empty"""
        adj[v1].remove(v2)
        if not adj[v1]:
            del adj[v1]
        adj[v2].remove(v1)
        if not adj[v2]:
            del adj[v2]

    # Keep extracting loops until the adjacency graph is empty
    while adj:
        start = next(iter(adj))
        loop = [start]
        prev = None
        current = start

        # Walk the adjacency graph to build a single closed loop starting and ending at 'start'
        while True:
            neighbors = adj[current]
            # Deterministic selection to produce stable output
            if prev is None:
                next_v = sorted(neighbors)[0]
            else:
                next_v = sorted(n for n in neighbors if n != prev)[0]

            _pop_edge(current, next_v)
            prev, current = current, next_v
            if current == start:
                break
            loop.append(current)

        loops.append(loop)

    return loops


def _add_bridges(loop: list[tuple[int,int]],
                min_bridges: int = 2,
                bridge_width: int = 1) -> list[list[tuple[int,int]]]:
    """
    Break the loop in `min_bridges` evenly spaced places by
    skipping `bridge_width` pixels.
    """
    n = len(loop)
    step = n // min_bridges
    keep = []
    i = 0
    while i < n:
        keep.append(loop[i])
        # when we’re at a bridge position, skip a few vertices
        if i % step == 0:
            i += bridge_width
        i += 1
    return keep