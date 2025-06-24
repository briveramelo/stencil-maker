import numpy as np
from PIL import Image


def masks_from_quantized(img_q: Image.Image):
    """
    For each colour index actually used, build a **bool ndarray mask**.

    Output shape: (len(palette), H, W)
    """
    data = np.array(img_q)  # 2-D array of palette indices
    masks = [(data == idx) for idx in sorted(set(data.flatten()))]
    return masks
