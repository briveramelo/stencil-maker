import numpy as np
from pathlib import Path
from typing import Iterable
import svgwrite

from src.models.models import RGB, RGBA


def masks_to_svgs(
    masks: Iterable,
    rgba_palette: list[RGBA],
    out_dir: Path,
    scale: int,
    base_filename: str,
):
    """
    Convert each mask into an SVG where every 'true' pixel becomes a <rect>.
    A more sophisticated implementation could flood-fill contiguous pixels
    and emit <path>s; this keeps dependencies light.
    """
    height, width = masks[0].shape
    size_px = (width * scale, height * scale)

    other_index = 1
    for mask, rgba in zip(masks, rgba_palette):
        rgb: RGB = rgba[:3]
        colour_label, other_index = _get_color_name(rgb, other_index)
        dwg = svgwrite.Drawing(
            filename=str(out_dir / f"{base_filename}_{colour_label}.svg"),
            size=size_px,
            viewBox=f"0 0 {width} {height}",
        )

        colour_hex = "#%02x%02x%02x" % rgb
        for x, y, w, h in _find_maximal_rectangles(mask):
            dwg.add(dwg.rect(insert=(x, y), size=(w, h), fill=colour_hex))

        dwg.save()


def _get_color_name(rgb: tuple[int, int, int], other_index: int) -> tuple[str, int]:
    """Return a human-friendly name for *rgb*.

    Values close to pure white or black are labeled accordingly to make the
    generated filenames predictable.
    """
    color_tolerance = 10
    if all(abs(c - 255) <= color_tolerance for c in rgb):
        return "white", other_index
    if all(abs(c) <= color_tolerance for c in rgb):
        return "black", other_index
    return f"color{other_index}", other_index + 1


def _find_maximal_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Find maximal rectangles in a binary mask. Returns (x, y, width, height) tuples."""
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D")

    height, width = mask.shape
    hist = [0] * width
    rectangles = []

    for y in range(height):
        for x in range(width):
            hist[x] = hist[x] + 1 if mask[y, x] else 0

        stack = []
        x = 0
        while x <= width:
            curr_height = hist[x] if x < width else 0
            if not stack or curr_height >= hist[stack[-1]]:
                stack.append(x)
                x += 1
            else:
                h = hist[stack.pop()]
                w = x if not stack else x - stack[-1] - 1
                x0 = stack[-1] + 1 if stack else 0
                rectangles.append((x0, y - h + 1, w, h))

    return rectangles