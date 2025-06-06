from pathlib import Path
from typing import Tuple
from PIL import Image
import numpy as np

RGBA = tuple[int, int, int, int]


def quantize_image(img_path: Path, k: int) -> Tuple[Image.Image, list[RGBA]]:
    """
    Load *img_path*, convert to RGBA, quantize to ≤k colours, return:
      - the quantized image in mode 'P'
      - the RGBA palette as a list of used colours (w/ real alpha)
    """
    img_rgba: Image.Image = Image.open(img_path).convert("RGBA")
    img_q: Image.Image = img_rgba.quantize(colors=k, method=Image.MEDIANCUT, dither=Image.NONE)

    pal = img_q.getpalette()
    full_rgb: list[tuple[int, int, int]] = [tuple(pal[i:i+3]) for i in range(0, len(pal), 3)]

    index_map = np.array(img_q)
    alpha_map = np.array(img_rgba.getchannel("A"))

    idx_to_alpha: dict[int, int] = {}
    for idx in np.unique(index_map):
        alpha_vals = alpha_map[index_map == idx]
        avg_alpha = int(np.mean(alpha_vals)) if alpha_vals.size > 0 else 255
        idx_to_alpha[idx] = avg_alpha

    used_idxs = sorted(set(index_map.flatten()))
    palette: list[RGBA] = [(*full_rgb[i], idx_to_alpha.get(i, 255)) for i in used_idxs]

    return img_q, palette
