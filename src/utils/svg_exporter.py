"""
Utility to convert boolean masks into minimal SVG stencil layers.

Features
--------
* One <path> per colour layer (compound path with holes via even‑odd fill rule).
* 4‑connected edge tracing → axis‑aligned, pixel‑perfect outlines (no diagonals).
* No extra dependencies beyond NumPy, OpenCV, and svgwrite.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import math
import numpy as np
import svgwrite

from src.models.models import RGB, RGBA


@dataclass(frozen=True)
class AlignmentCutConfig:
    """Configuration for «+»-shaped alignment cuts."""

    height: float
    thickness: float

    @property
    def padding(self) -> float:
        """Extra canvas padding required to fit the cut fully outside the art."""

        return self.height + self.thickness


@dataclass(frozen=True)
class CompositeGridConfig:
    """Configuration for composite SVG grids that merge blinds outputs."""

    columns: int

    def __post_init__(self) -> None:
        if self.columns <= 0:
            msg = "columns must be ≥ 1 for CompositeGridConfig"
            raise ValueError(msg)


@dataclass
class _CompositeCell:
    """In-memory representation of a single cell in the composite grid."""

    loops: list[list[tuple[float, float]]]
    rgb: RGB


def masks_to_svgs(
    masks: Iterable[np.ndarray],
    rgba_palette: list[RGBA],
    out_dir: Path,
    scale: int,
    base_filename: str,
    alignment_cuts: AlignmentCutConfig | None = None,
    composite_grid: CompositeGridConfig | None = None,
) -> None:
    """
    Export each mask as three SVG files using a "persian blinds" split:
      - full: all pixels
      - blinds1: rows 1-2, 5-6, 9-10, ...
      - blinds2: rows 3-4, 7-8, 11-12, ...

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
    composite_grid
        Optional grid specification to write composite SVGs that combine the
        blinds outputs for every mask.
    """
    masks = list(masks)
    out_dir.mkdir(parents=True, exist_ok=True)

    height, width = masks[0].shape
    padding = alignment_cuts.padding if alignment_cuts else 0.0
    doc_width = width + 2 * padding
    doc_height = height + 2 * padding
    doc_px_size = (doc_width * scale, doc_height * scale)

    composite_cells: dict[str, list[_CompositeCell]] | None
    if composite_grid:
        composite_cells = {"blinds1": [], "blinds2": []}
    else:
        composite_cells = None

    other_index = 1
    for mask, rgba in zip(masks, rgba_palette):
        rgb: RGB = rgba[:3]
        color_label, other_index = _get_color_name(rgb, other_index)

        # Create three variants: full, blinds1, blinds2
        blinds1 = np.zeros_like(mask, dtype=bool)
        blinds2 = np.zeros_like(mask, dtype=bool)
        # Persians blinds logic: group rows in pairs: (0,1)->blinds1, (2,3)->blinds2, (4,5)->blinds1, ...
        for row in range(height):
            if mask[row].any():
                block = (row // 2) % 2
                if block == 0:
                    blinds1[row, :] = mask[row, :]
                else:
                    blinds2[row, :] = mask[row, :]

        variants = [("full", mask), ("blinds1", blinds1), ("blinds2", blinds2)]

        for suffix, variant in variants:
            dwg = svgwrite.Drawing(
                filename=str(out_dir / f"{base_filename}_{color_label}_{suffix}.svg"),
                size=doc_px_size,
                viewBox=f"0 0 {doc_width} {doc_height}",
            )

            loops = _offset_loops(
                _trace_contours_as_loops(variant), padding, padding
            )
            if composite_cells is not None and suffix in composite_cells:
                loops_copy = [loop.copy() for loop in loops]
                composite_cells[suffix].append(
                    _CompositeCell(loops=loops_copy, rgb=rgb)
                )
            alignment_loops: Sequence[Sequence[tuple[float, float]]]
            if alignment_cuts:
                alignment_loops = _alignment_cut_loops(
                    alignment_cuts, width, height, padding
                )
            else:
                alignment_loops = []

            all_loops: list[Sequence[tuple[float, float]]] = list(loops)
            all_loops.extend(alignment_loops)

            if all_loops:
                path_d = _loops_to_svg_path(all_loops)
                dwg.add(
                    dwg.path(
                        d=path_d,
                        fill=f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}",
                        fill_rule="evenodd",
                    )
                )
            dwg.save()

    if composite_grid and composite_cells is not None:
        _write_composite_svgs(
            composite_cells,
            doc_width,
            doc_height,
            scale,
            out_dir,
            base_filename,
            composite_grid,
        )


def _write_composite_svgs(
    cells_by_suffix: dict[str, list[_CompositeCell]],
    cell_width: float,
    cell_height: float,
    scale: int,
    out_dir: Path,
    base_filename: str,
    grid_config: CompositeGridConfig,
) -> None:
    """Write composite SVGs that arrange blinds outputs on a fixed grid."""

    columns = grid_config.columns
    for suffix, cells in cells_by_suffix.items():
        if not cells:
            continue

        total_cells = len(cells)
        rows = math.ceil(total_cells / columns)
        total_width = columns * cell_width
        total_height = rows * cell_height
        doc_px_size = (total_width * scale, total_height * scale)

        dwg = svgwrite.Drawing(
            filename=str(out_dir / f"{base_filename}_{suffix}_composite.svg"),
            size=doc_px_size,
            viewBox=f"0 0 {total_width} {total_height}",
        )

        for index, cell in enumerate(cells):
            row = index // columns
            col = index % columns
            offset_x = col * cell_width
            offset_y = row * cell_height

            if not cell.loops:
                continue

            loops = _offset_loops(cell.loops, offset_x, offset_y)
            if not loops:
                continue

            path_d = _loops_to_svg_path(loops)
            if not path_d:
                continue

            rgb = cell.rgb
            dwg.add(
                dwg.path(
                    d=path_d,
                    fill=f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}",
                    fill_rule="evenodd",
                )
            )

        stroke_kwargs = {"stroke": "#000000", "stroke_width": 0, "fill": "none"}
        for col in range(columns + 1):
            x = col * cell_width
            dwg.add(dwg.line(start=(x, 0), end=(x, total_height), **stroke_kwargs))
        for row in range(rows + 1):
            y = row * cell_height
            dwg.add(dwg.line(start=(0, y), end=(total_width, y), **stroke_kwargs))

        dwg.save()


def _get_color_name(rgb: RGB, other_index: int) -> tuple[str, int]:
    """Return a predictable colour label for filenames."""
    tol = 10
    if all(abs(c - 255) <= tol for c in rgb):
        return "white", other_index
    if all(c <= tol for c in rgb):
        return "black", other_index
    return f"color{other_index}", other_index + 1


def _trace_contours_as_loops(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Convert a binary mask into ordered loops describing its borders."""

    return _trace_edges(mask)


def _offset_loops(
    loops: Iterable[Iterable[tuple[int, int]]],
    offset_x: float,
    offset_y: float,
) -> list[list[tuple[float, float]]]:
    """Translate every vertex in *loops* by the provided offsets."""

    translated: list[list[tuple[float, float]]] = []
    for loop in loops:
        translated.append([(x + offset_x, y + offset_y) for x, y in loop])
    return translated


def _loops_to_svg_path(loops: Iterable[Sequence[tuple[float, float]]]) -> str:
    """Convert vertex loops into a single SVG path string."""

    sub_paths: list[str] = []
    for loop in loops:
        points = list(loop)
        if not points:
            continue
        pieces = [f"M {points[0][0]} {points[0][1]}"]
        pieces.extend(f"L {x} {y}" for x, y in points[1:])
        pieces.append("Z")
        sub_paths.append(" ".join(pieces))
    return " ".join(sub_paths)


def _alignment_cut_loops(
    config: AlignmentCutConfig,
    width: int,
    height: int,
    padding: float,
) -> list[list[tuple[float, float]]]:
    """Return loops describing «+»-shaped alignment cuts around the stencil."""

    if config.height <= 0 or config.thickness <= 0:
        return []

    half_height = config.height / 2
    half_thickness = config.thickness / 2
    offset = half_height + half_thickness

    centers = [
        (padding - offset, padding - offset),  # top-left
        (padding + width + offset, padding - offset),  # top-right
        (padding - offset, padding + height + offset),  # bottom-left
        (padding + width + offset, padding + height + offset),  # bottom-right
    ]

    loops: list[list[tuple[float, float]]] = []
    for cx, cy in centers:
        loops.append(
            [
                (cx - half_thickness, cy - half_height),
                (cx + half_thickness, cy - half_height),
                (cx + half_thickness, cy - half_thickness),
                (cx + half_height, cy - half_thickness),
                (cx + half_height, cy + half_thickness),
                (cx + half_thickness, cy + half_thickness),
                (cx + half_thickness, cy + half_height),
                (cx - half_thickness, cy + half_height),
                (cx - half_thickness, cy + half_thickness),
                (cx - half_height, cy + half_thickness),
                (cx - half_height, cy - half_thickness),
                (cx - half_thickness, cy - half_thickness),
            ]
        )

    return loops


def _trace_edges(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """
    Trace 4‑connected pixel edges and return ordered loops.

    Returns a list of loops; each loop is a list of integer (x, y) vertices on
    pixel boundaries.  Even‑odd fill handles holes automatically.
    """
    mask = mask.astype(bool)
    h, w = mask.shape
    adj: dict[tuple[int, int], set[tuple[int, int]]] = {}

    def _add_edge(v1: tuple[int, int], v2: tuple[int, int]) -> None:
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

    loops: list[list[tuple[int, int]]] = []

    def _pop_edge(v1: tuple[int, int], v2: tuple[int, int]) -> None:
        adj[v1].remove(v2)
        if not adj[v1]:
            del adj[v1]
        adj[v2].remove(v1)
        if not adj[v2]:
            del adj[v2]

    while adj:
        start = next(iter(adj))
        loop = [start]
        prev = None
        current = start

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
