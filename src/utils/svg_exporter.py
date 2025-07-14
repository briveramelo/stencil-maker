from pathlib import Path
from typing import Iterable
import svgwrite

from src.models.models import RGBA


_COLOUR_TOLERANCE = 5


def _colour_name(rgb: tuple[int, int, int], other_index: int) -> tuple[str, int]:
    """Return a human-friendly name for *rgb*.

    Values close to pure white or black are labeled accordingly to make the
    generated filenames predictable.
    """

    if all(abs(c - 255) <= _COLOUR_TOLERANCE for c in rgb):
        return "white", other_index
    if all(abs(c) <= _COLOUR_TOLERANCE for c in rgb):
        return "black", other_index
    return f"color{other_index}", other_index + 1


def masks_to_svgs(
    masks: Iterable,
    palette: list[RGBA],
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
    for mask, rgba in zip(masks, palette):
        colour_label, other_index = _colour_name(rgba[:3], other_index)
        dwg = svgwrite.Drawing(
            filename=str(out_dir / f"{base_filename}_{colour_label}.svg"),
            size=size_px,
            viewBox=f"0 0 {width} {height}",  # avoids bloated coordinate space
        )
        # dwg.add(dwg.rect(insert=(0, 0), size=(width, height), fill="none"))  # bounding box

        colour_hex = "#%02x%02x%02x" % rgba[:3]
        for y, row in enumerate(mask):
            for x, on in enumerate(row):
                if on:
                    dwg.add(dwg.rect(insert=(x, y), size=(1, 1), fill=colour_hex))

        dwg.save()
