import cv2
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
        color_label, other_index = _get_color_name(rgb, other_index)
        dwg = svgwrite.Drawing(
            filename=str(out_dir / f"{base_filename}_{color_label}.svg"),
            size=size_px,
            viewBox=f"0 0 {width} {height}",
        )

        color_hex = "#%02x%02x%02x" % rgb
        path_d = _trace_contours_as_svg_paths(mask)
        if path_d:
            dwg.add(dwg.path(d=path_d, fill=color_hex, fill_rule="evenodd"))

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


def _trace_contours_as_svg_paths(mask: np.ndarray) -> str:
    """
    Convert a binary mask into a single SVG‐compatible path string.

    The resulting path:
      • Uses only one <path> element (compound path with “holes”).
      • Emits commands with **spaces** (no commas) so svgwrite’s validator accepts it.
      • Closes every contour with “Z” and relies on the even-odd fill rule.
    """
    # Convert mask to uint8 (0 or 255) for OpenCV
    mask_u8 = (mask.astype(np.uint8)) * 255

    # Find outer contours and holes in one pass
    contours, hierarchy = cv2.findContours(
        mask_u8,
        mode=cv2.RETR_CCOMP,          # get contour hierarchy (outer + holes)
        method=cv2.CHAIN_APPROX_NONE  # keep every pixel for pixel-perfect edges
    )

    if not contours or hierarchy is None:
        return ""

    hierarchy = hierarchy[0]  # shape (N, 4): [next, prev, first_child, parent]
    path_parts: list[str] = []

    def _contour_to_path(cnt: np.ndarray) -> str:
        """Return an SVG sub-path (“M … L … Z”) for one contour."""
        pts = cnt.reshape(-1, 2)  # (N, 2)
        if pts.size == 0:
            return ""
        pieces = [f"M {pts[0][0]} {pts[0][1]}"]
        for x, y in pts[1:]:
            pieces.append(f"L {x} {y}")
        pieces.append("Z")
        return " ".join(pieces)

    for idx, (cnt, h) in enumerate(zip(contours, hierarchy)):
        if h[3] == -1:  # this is a top-level (outer) contour
            # Start with the outer contour
            sub_path = _contour_to_path(cnt)

            # Append any child contours (holes) to the same compound path
            for child_idx, h_child in enumerate(hierarchy):
                if h_child[3] == idx:  # parent equals current outer contour
                    sub_path += " " + _contour_to_path(contours[child_idx])

            path_parts.append(sub_path)

    # Combine all outer-with-holes sub-paths into one string
    return " ".join(path_parts)


def _trace_edges(mask: np.ndarray) -> list[list[tuple[int,int]]]:
    """Return a list of outer-with-holes polylines; each polyline is a list of (x,y) vertices on pixel *edges*."""
    h, w = mask.shape
    visited = set()
    paths = []

    # Helper: find the first foreground pixel that has an exposed edge
    def find_start():
        for y in range(h):
            for x in range(w):
                if mask[y, x] and (x, y, 0) not in visited:  # 0 = heading east
                    return x, y
        return None

    # Right-hand-rule walker
    DIRS = [(1,0), (0,1), (-1,0), (0,-1)]  # E,S,W,N ; indices 0..3
    while (start := find_start()) is not None:
        x, y = start
        dir_idx = 3  # we arrive "from the north" so we start heading east
        path = []
        while True:
            path.append((x, y))
            for i in range(4):
                ndir = (dir_idx + 3 + i) % 4  # turn right until we see an edge
                dx, dy = DIRS[ndir]
                left_x, left_y = x + dx, y + dy
                if 0 <= left_x < w and 0 <= left_y < h and mask[left_y, left_x]:
                    # Move along the border; mark entrance direction visited
                    visited.add((x, y, ndir))
                    x, y = left_x, left_y
                    dir_idx = ndir
                    break
            else:
                break  # isolated pixel
            if (x, y) == start and dir_idx == 3:
                break  # loop closed
        paths.append(path)
    return paths