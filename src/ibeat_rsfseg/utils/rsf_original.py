import logging

import numpy as np
import skimage
from dipy.segment.mask import median_otsu
import scipy.ndimage as ndi


def renal_sinus_fat(fat, kidneys):
    """Compute renal sinus fat mask via convex hull of the kidney mask.

    fat: numpy array, fat-channel Dixon volume
    kidneys: numpy array, kidney label mask (1=left, 2=right kidney)
    """
    rsf = np.zeros(kidneys.shape)
    fat_mask = _median_otsu_2d(fat, median_radius=1, numpass=1)

    for kidney in [1, 2]:
        mask = (kidneys == kidney).astype(int)

        if not mask.any():
            continue

        try:
            kidney_hull = _convex_hull_image_3d(mask)
        except Exception as e:
            logging.error(f"Convex hull failed for label {kidney}: {e}")
            continue

        # Sinus = fat within the convex hull area
        sinus_fat = fat_mask * kidney_hull

        if not sinus_fat.any():
            continue

        sinus_fat_largest = _extract_largest_cluster_3d(sinus_fat)
        rsf[sinus_fat_largest] = kidney

    return rsf


def _median_otsu_2d(array, **kwargs):
    mask = np.empty(array.shape)
    for z in range(array.shape[2]):
        image = np.squeeze(array[:, :, z])
        _, mask[:, :, z] = median_otsu(image, **kwargs)
    return mask


def _convex_hull_image_3d(array, **kwargs):

    volume = np.around(array)
    hull = np.zeros_like(volume, dtype=bool)
    for z in range(volume.shape[2]):
        slice_ = volume[:, :, z]
        if not slice_.any():
            continue
        hull[:, :, z] = skimage.morphology.convex_hull_image(slice_, **kwargs)
    return hull


def _extract_largest_cluster_3d(array):
    structure = np.ones((3, 3, 3)) 
    label_img, cnt = ndi.label(array, structure=structure)
    sizes = ndi.sum(array, label_img, index=range(1, cnt + 1))
    max_label = np.argmax(sizes) + 1
    return label_img == max_label