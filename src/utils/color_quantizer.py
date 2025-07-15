from pathlib import Path
from typing import Tuple
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

from src.models.models import RGBA, RGB


def quantize_image(img_path: Path, k: int, alpha_threshold: int = 10) -> Tuple[np.ndarray, list[RGBA]]:
    """
    Load *img_path*, convert to RGBA, quantize only opaque pixels to exactly k colours, return:
      - the 2D label map (index map) of shape (H, W), where each value is 0..k-1 for opaque pixels, -1 for transparent
      - the RGBA palette as a list of k used colours (w/ real alpha)
    Transparent (or nearly transparent) pixels are ignored in quantization and marked as -1 in the label map.
    """
    img_rgba = Image.open(img_path).convert("RGBA")
    arr = np.array(img_rgba)
    # Mask for opaque pixels
    opaque_mask = arr[..., 3] >= alpha_threshold
    opaque_pixels = arr[opaque_mask][..., :3]  # shape (N, 3)
    if len(opaque_pixels) == 0:
        raise ValueError("No opaque pixels found in image.")
    # KMeans quantization
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(opaque_pixels)
    palette = kmeans.cluster_centers_.astype(np.uint8)
    # Build RGBA palette (all opaque)
    rgba_palette: list[RGBA] = [(int(r), int(g), int(b), 255) for r, g, b in palette]
    # Build 2D label map, -1 for transparent
    label_map = np.full(arr.shape[:2], fill_value=-1, dtype=np.int32)
    label_map[opaque_mask] = labels
    return label_map, rgba_palette
