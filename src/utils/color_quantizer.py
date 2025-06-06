from pathlib import Path

import numpy as np
from PIL import Image


def quantize_image(img_path: Path, k: int):
    """
    Load *img_path*, convert to RGBA, and Pillow-quantize it
    to **≤ k** distinct colours (with dithering off).

    Returns
    -------
    img_q   : PIL.Image.Image – the quantised image (mode 'P')
    palette : list[tuple[int, int, int, int]] – RGBA colours actually used
    """
    img = Image.open(img_path).convert("RGBA")
    img_q = img.quantize(colors=k, method=Image.MEDIANCUT, dither=Image.NONE)

    # Pillow stores the palette as an RGB array – extract only colours present
    pal = img_q.getpalette()  # flat list length 768
    full_rgba = [
        (*pal[i : i + 3], 255)  # alpha full-opaque; PNG has original alpha in mask
        for i in range(0, len(pal), 3)
    ]
    used_idxs = sorted(set(np.array(img_q)))
    palette = [full_rgba[i] for i in used_idxs]
    return img_q, palette
