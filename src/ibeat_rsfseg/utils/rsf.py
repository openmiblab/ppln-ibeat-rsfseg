import os
import logging

import numpy as np
import skimage
import matplotlib
matplotlib.use('Agg')  # no display needed, only saving to file



def renal_sinus_fat(fat, water, kidneys, fixed_threshold=0.5):
    """Compute renal sinus fat mask via convex hull of the kidney mask.

    fat: numpy array, fat-channel Dixon volume
    water: numpy array, water-channel Dixon volume, same shape as
        `fat`. Used together with `fat` to compute the fat fraction map
        (fat / (fat + water)); the fat/non-fat classification is done on
        that fraction map, not on the raw fat channel.
    kidneys: numpy array, kidney label mask (1=left, 2=right kidney)
    fixed_threshold: float, threshold on the fat fraction map: voxels
        with fat_fraction > fixed_threshold are considered fat (no more
        Otsu/median_otsu).
    

    """
    rsf = np.zeros(kidneys.shape)

    # Fat fraction map: all thresholding is done on this, instead of
    # on the raw fat channel.
    fat_fraction = _fat_fraction_map(fat, water)

    for kidney in [1, 2]:
        side = 'left' if kidney == 1 else 'right'
        mask = (kidneys == kidney).astype(int)

        if not mask.any():
            continue

        try:
            kidney_hull = _convex_hull_image_3d(mask)
        except Exception as e:
            logging.error(f"Convex hull failed fo label {kidney}: {e}")
            continue

        # Fat/non-fat classification: fixed threshold on the fat-fraction
        # map, per kidney, within a crop around the convex hull.
        fat_mask = _fixed_threshold_mask(
            fat_fraction, kidney_hull, threshold=fixed_threshold)

       
        sinus_fat = fat_mask * kidney_hull     
        rsf = sinus_fat

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


def _fixed_threshold_mask(fat_fraction, threshold=0.5):
    """Fixed-threshold fat mask on the fraction map: voxels with
    fat_fraction > threshold are considered fat. No median_otsu.

    """
    fat_mask = np.zeros(fat_fraction.shape, dtype=float)

    fat_mask[fat_fraction > threshold] = 1.0
    return fat_mask


def _convex_hull_image_3d(array, **kwargs):

    volume = np.around(array)
    hull = np.zeros_like(volume, dtype=bool)
    for z in range(volume.shape[2]):
        slice_ = volume[:, :, z]
        if not slice_.any():
            continue
        hull[:, :, z] = skimage.morphology.convex_hull_image(slice_, **kwargs)
    return hull




