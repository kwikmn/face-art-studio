"""Crop-contained opacity masks for the protected face replacement pipeline."""

import numpy as np


def face_mask(size, jaw_feather=0.6):
    """Smooth elliptical edge with a fully opaque central facial region.

    Feather width increases only toward the lower cheeks/chin. The support
    ends inside the crop, so warping with a zero border cannot expose a cutoff.
    All distances are normalized so the edge scales with face size.
    """
    if size < 16 or not np.isfinite(jaw_feather) or not 0 <= jaw_feather <= 1:
        raise ValueError("Invalid face mask size or feathering")
    y, x = np.mgrid[:size, :size].astype(np.float32) / (size - 1)
    radius = np.sqrt(((x - 0.5) / 0.49) ** 2 + ((y - 0.5) / 0.49) ** 2)
    lower = np.clip((y - 0.45) / 0.30, 0, 1)
    lower = lower * lower * (3 - 2 * lower)
    width = 0.16 + lower * (0.02 + 0.16 * jaw_feather)
    fade = np.clip((1 - radius) / width, 0, 1)
    fade = fade * fade * (3 - 2 * fade)
    return np.rint(fade * 255).astype(np.uint8)
