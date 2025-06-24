from pathlib import Path
from typing import Iterable
import svgwrite

from src.models.models import RGBA


def masks_to_svgs(
    masks: Iterable,
    palette: list[RGBA],
    out_dir: Path,
    scale: int
):
    """
    Convert each mask into an SVG where every 'true' pixel becomes a <rect>.
    A more sophisticated implementation could flood-fill contiguous pixels
    and emit <path>s; this keeps dependencies light.
    """
    height, width = masks[0].shape
    size_px = (width * scale, height * scale)

    for index, (mask, rgba) in enumerate(zip(masks, palette), start=1):
        dwg = svgwrite.Drawing(
            filename=str(out_dir / f"layer_{index:02}.svg"),
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
