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

def _make_bridges(loop: Loop, img_w: int, img_h: int, span: float = 2.0, half_thickness: float = 0.5) -> list[str]:
    """
    Generate two small rectangular bridge sub‑paths (left and right) for an
    interior hole so it stays attached to the surrounding stencil.

    Bridges are only added when the loop is fully inside the image bounds,
    which indicates that the loop represents a hole rather than the outer
    silhouette.

    Parameters
    ----------
    loop : Loop
        Ordered list of (x, y) vertices describing the polygon.
    img_w, img_h : int
        Overall mask dimensions — used to skip outer border loops.
    span : float, default 2.0
        Horizontal length, in pixels, that each bridge protrudes.
    half_thickness : float, default 0.5
        Half the vertical thickness of each bridge (total thickness is
        2 × half_thickness).

    Returns
    -------
    list[str]
        Zero, one, or two SVG sub‑path strings to be concatenated with the
        loop’s main path.
    """
    xs = [p[0] for p in loop]
    ys = [p[1] for p in loop]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Skip loops that touch the image border – they are outer outlines.
    if min_x == 0 or max_x == img_w or min_y == 0 or max_y == img_h:
        return []

    y_mid = (min_y + max_y) / 2
    bridges = []

    # Left‑side bridge
    bridges.append(
        " ".join(
            [
                f"M {min_x} {y_mid - half_thickness}",
                f"L {min_x - span} {y_mid - half_thickness}",
                f"L {min_x - span} {y_mid + half_thickness}",
                f"L {min_x} {y_mid + half_thickness}",
                "Z",
            ]
        )
    )

    # Right‑side bridge
    bridges.append(
        " ".join(
            [
                f"M {max_x} {y_mid - half_thickness}",
                f"L {max_x + span} {y_mid - half_thickness}",
                f"L {max_x + span} {y_mid + half_thickness}",
                f"L {max_x} {y_mid + half_thickness}",
                "Z",
            ]
        )
    )

    return bridges

# ----------------------------------------------------------------------
#  Polygon utilities for “island” detection
# ----------------------------------------------------------------------
def _point_in_loop(pt: tuple[float, float], loop: Loop) -> bool:
    """
    Ray‑crossing test (even‑odd rule) to see whether point *pt* lies strictly
    inside *loop*.  Loops are axis‑aligned but may be concave.
    """
    x, y = pt
    inside = False
    n = len(loop)
    for i in range(n):
        x1, y1 = loop[i]
        x2, y2 = loop[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-6) + x1
        ):
            inside = not inside
    return inside


def _compute_loop_depths(loops: list[Loop]) -> list[int]:
    """
    For every loop, count how many *other* loops contain a point inside it.
    The count is the nesting depth:
        0 → outermost outline
        1 → simple hole
        2 → island inside that hole
        …
    """
    depths: list[int] = []
    for i, loop in enumerate(loops):
        # Use polygon centroid as a guaranteed interior point
        xs, ys = zip(*loop)
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        depth = sum(
            1 for j, other in enumerate(loops) if j != i and _point_in_loop((cx, cy), other)
        )
        depths.append(depth)
    return depths


def _trace_contours_as_svg_paths(mask: np.ndarray) -> str:
    """
    Convert a binary mask into a single SVG path string (compound with holes).

    Uses a custom 4‑connected border trace so that all segments are axis‑aligned
    and lie exactly on pixel boundaries — perfect for cutters.
    """
    loops = _trace_edges(mask)
    depths = _compute_loop_depths(loops)
    h, w = mask.shape  # For bridge placement
    if not loops:
        return ""

    sub_paths = []
    for loop, depth in zip(loops, depths):
        pieces = [f"M {loop[0][0]} {loop[0][1]}"] # Begin the sub‑path with a Move‑to (M) command at the first vertex
        pieces.extend(f"L {x} {y}" for x, y in loop[1:]) # Append Line‑to (L) commands for every remaining vertex
        pieces.append("Z") # Close the current sub‑path (Z) to finish the loop
        sub_paths.append(" ".join(pieces))

        # Add bridges **only** when loop is an “island”
        # (even nesting depth ≥ 2 ⇒ filled region inside a hole)
        if depth >= 2 and depth % 2 == 0:
            bridges = _make_bridges(loop, w, h, span=1, half_thickness=0.25)
            sub_paths.extend(bridges)

    return " ".join(sub_paths)
