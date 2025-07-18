import numpy as np
from PIL import Image


def masks_from_quantized(img_q: Image.Image, rgba_palette):
    """Build boolean masks for each used colour index in *img_q*.

    Transparent pixels (alpha == 0) are excluded from all masks so that
    invisible areas do not end up in any stencil layer.

    Output shape: (len(palette), H, W)
    """
    data = np.array(img_q)  # 2‑D array of palette indices
    alpha = np.array(img_q.convert("RGBA").getchannel("A"))

    masks = [
        ((data == idx) & (alpha > 0)) for idx in sorted(set(data.flatten()))
    ]
    return masks
