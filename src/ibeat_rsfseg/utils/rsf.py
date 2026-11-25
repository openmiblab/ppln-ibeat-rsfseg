import logging

import numpy as np
import skimage.morphology


def renal_sinus_fat(fat, water, kidneys, fixed_threshold=0.5, **kwargs):
    """Compute renal sinus fat mask via convex hull of the kidney mask.

    fat: numpy array, fat-channel Dixon volume
    water: numpy array, water-channel Dixon volume, same shape as `fat`.
        Used together with `fat` to compute the fat fraction map
        (fat / (fat + water)).
    kidneys: numpy array, kidney label mask (1=left, 2=right kidney)
    fixed_threshold: float, voxels with fat_fraction > fixed_threshold
        inside the convex hull are considered sinus fat.
    **kwargs: accepted for compatibility with older pipeline calls
        (bounded, max_dilation, pole_cut, ...) but ignored.

    Returns an array with label 1 (left) and 2 (right) for sinus fat.
    """
    rsf = np.zeros(kidneys.shape, dtype=np.uint8)
    fat_fraction = _fat_fraction_map(fat, water)

    for kidney in [1, 2]:
        mask = (kidneys == kidney)
        if not mask.any():
            continue

        try:
            kidney_hull = _convex_hull_image_3d(mask)
        except Exception as e:
            logging.error(f"Convex hull failed for label {kidney}: {e}")
            continue

        sinus_fat = kidney_hull & (fat_fraction > fixed_threshold)
        rsf[sinus_fat] = kidney

    return rsf


def _fat_fraction_map(fat, water, eps=1e-6):
    """Fat fraction map: fat / (fat + water), clipped to [0, 1].

    eps prevents division by zero in background voxels where both fat
    and water are ~0.
    """
    fat = fat.astype(float)
    water = water.astype(float)
    denom = fat + water + eps
    return np.clip(fat / denom, 0.0, 1.0)


def _convex_hull_image_3d(array, **kwargs):
    """Convex hull computed slice by slice along the z-axis."""
    volume = np.around(array).astype(bool)
    hull = np.zeros(volume.shape, dtype=bool)
    for z in range(volume.shape[2]):
        slice_ = volume[:, :, z]
        if not slice_.any():
            continue
        hull[:, :, z] = skimage.morphology.convex_hull_image(slice_, **kwargs)
    return hull