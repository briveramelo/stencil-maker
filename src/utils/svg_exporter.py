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
    """Configuration describing how blinds variants are tiled into a grid."""

    rows: int
    cols: int
    stroke_width: float = 0.1

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise ValueError("CompositeGridConfig.rows must be positive")
        if self.cols <= 0:
            raise ValueError("CompositeGridConfig.cols must be positive")
        if self.stroke_width <= 0:
            raise ValueError("CompositeGridConfig.stroke_width must be positive")


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
        When provided, create additional «_composite» files for blinds variants
        laid out in a *rows*×*cols* grid with thin divider lines.
    """
    masks = list(masks)
    out_dir.mkdir(parents=True, exist_ok=True)

    height, width = masks[0].shape
    padding = alignment_cuts.padding if alignment_cuts else 0.0
    doc_width = width + 2 * padding
    doc_height = height + 2 * padding
    doc_px_size = (doc_width * scale, doc_height * scale)

    other_index = 1
    composite_sources: dict[
        str, list[tuple[Sequence[Sequence[tuple[float, float]]], RGB]]
    ] = {}
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

            if composite_grid and suffix.startswith("blinds"):
                stored_loops: list[Sequence[tuple[float, float]]] = [
                    tuple(loop) for loop in loops
                ]
                composite_sources.setdefault(suffix, []).append((stored_loops, rgb))

    if composite_grid and composite_sources:
        _write_composite_svgs(
            composite_sources,
            out_dir,
            base_filename,
            doc_width,
            doc_height,
            scale,
            composite_grid,
        )


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


def _write_composite_svgs(
    composites: dict[str, list[tuple[Sequence[Sequence[tuple[float, float]]], RGB]]],
    out_dir: Path,
    base_filename: str,
    cell_width: float,
    cell_height: float,
    scale: int,
    grid: CompositeGridConfig,
) -> None:
    """Write composite SVGs that tile blinds variants into a grid."""

    total_cells = grid.rows * grid.cols

    for suffix, entries in composites.items():
        if not entries:
            continue
        if len(entries) > total_cells:
            raise ValueError(
                "Composite grid %dx%d is too small for %d layers"
                % (grid.rows, grid.cols, len(entries))
            )

        comp_width = cell_width * grid.cols
        comp_height = cell_height * grid.rows
        comp_px_size = (comp_width * scale, comp_height * scale)

        dwg = svgwrite.Drawing(
            filename=str(out_dir / f"{base_filename}_{suffix}_composite.svg"),
            size=comp_px_size,
            viewBox=f"0 0 {comp_width} {comp_height}",
        )

        # Draw outer border
        dwg.add(
            dwg.rect(
                insert=(0, 0),
                size=(comp_width, comp_height),
                fill="none",
                stroke="#000000",
                stroke_width=grid.stroke_width,
            )
        )

        # Draw internal grid lines
        for col in range(1, grid.cols):
            x = col * cell_width
            dwg.add(
                dwg.line(
                    start=(x, 0),
                    end=(x, comp_height),
                    stroke="#000000",
                    stroke_width=grid.stroke_width,
                )
            )
        for row in range(1, grid.rows):
            y = row * cell_height
            dwg.add(
                dwg.line(
                    start=(0, y),
                    end=(comp_width, y),
                    stroke="#000000",
                    stroke_width=grid.stroke_width,
                )
            )

        for index, (loops, rgb) in enumerate(entries):
            row = index // grid.cols
            col = index % grid.cols
            offset_x = col * cell_width
            offset_y = row * cell_height
            translated = _offset_loops(loops, offset_x, offset_y)
            if not translated:
                continue
            path_d = _loops_to_svg_path(translated)
            dwg.add(
                dwg.path(
                    d=path_d,
                    fill=f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}",
                    fill_rule="evenodd",
                )
            )

        dwg.save()


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
