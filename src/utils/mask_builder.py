import numpy as np


def masks_from_quantized(label_map: np.ndarray, rgba_palette: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
    """Build boolean masks for each used colour index in *label_map*.

    Transparent pixels (alpha == 0) are excluded from all masks so that
    invisible areas do not end up in any stencil layer.

    Output shape: (len(palette), H, W)
    """
    data = label_map  # 2‑D array of palette indices
    # Build alpha mask from palette's alpha channel
    # Create an alpha array matching data shape, where each pixel's alpha is taken from the palette
    alpha = np.array([[rgba_palette[idx][3] for idx in row] for row in data])

    masks = [
        ((data == idx) & (alpha > 0)) for idx in sorted(set(data.flatten()))
    ]
    return masks
