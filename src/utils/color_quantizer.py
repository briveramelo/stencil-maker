from pathlib import Path
from typing import Tuple
from PIL import Image
import numpy as np
from PIL.Image import Quantize, Dither

from src.models.models import RGBA, RGB


def quantize_image(img_path: Path, k: int) -> Tuple[Image.Image, list[RGBA]]:
    """
    Load *img_path*, convert to RGBA, quantize to ≤k colours, return:
      - the quantized image in mode 'P'
      - the RGBA palette as a list of used colours (w/ real alpha)
    """
    img_rgba = Image.open(img_path).convert("RGBA")
    img_q = img_rgba.quantize(colors=k, method=Quantize.FASTOCTREE, dither=Dither.NONE)

    # 1) Grab the *raw* palette (flat list of ints), assert it's present
    raw_palette = img_q.getpalette()
    if raw_palette is None:
        raise RuntimeError("Expected a palette on quantized image, but got None")

    # 2) Chunk into RGB triplets
    full_rgb: list[RGB] = [(raw_palette[i], raw_palette[i + 1], raw_palette[i + 2]) for i in range(int(len(raw_palette) / 3))]

    # 3) Build an index→alpha map from the original RGBA image
    index_map = np.array(img_q)
    alpha_map = np.array(img_rgba.getchannel("A"))

    idx_to_alpha: dict[int, int] = {}
    for idx in np.unique(index_map):
        alphas = alpha_map[index_map == idx]
        idx_to_alpha[idx] = int(alphas.mean()) if alphas.size > 0 else 255

    # 4) Only keep the indices actually used in the image
    used_indexes = sorted(set(index_map.flatten()))

    # 5) Build your final RGBA palette list
    rgba_palette: list[RGBA] = [(full_rgb[i][0], full_rgb[i][1], full_rgb[i][2], idx_to_alpha.get(i, 255)) for i in used_indexes]

    return img_q, rgba_palette
