"""Fat-fraction and renal-sinus-specific features.
"""

import logging

import numpy as np
import scipy.ndimage as ndi
from scipy import stats


def fat_fraction_map(fat, water):
    """Fat fraction in percent: 100 * fat / (fat + water). NaN where fat + water <= 0."""
    fat = np.asarray(fat, dtype=float)
    water = np.asarray(water, dtype=float)
    total = fat + water
    ff = np.full(total.shape, np.nan)
    np.divide(100.0 * fat, total, out=ff, where=total > 0)
    return np.clip(ff, 0.0, 100.0)  # NaN is preserved by clip


def check_same_grid(arr, src_affine, dst_affine, dst_shape, what, atol=1e-3):
    """Verify that arr is on exactly the same grid as the mask (shape and affine).

    No resampling is done: when the grids differ a ValueError is raised, so a
    mismatch is reported instead of silently producing wrong numbers.
    """
    if tuple(arr.shape) != tuple(dst_shape):
        raise ValueError(
            f"{what}: shape {tuple(arr.shape)} differs from mask shape {tuple(dst_shape)}"
        )
    if not np.allclose(src_affine, dst_affine, atol=atol):
        raise ValueError(f"{what}: affine differs from the mask affine")
    return arr


def rsf_features(roi, mask, spacing, ff=None, kidney_mask=None):
    """Features for one sinus fat ROI.

    mask        : 3D bool/0-1 array (the ROI)
    spacing     : voxel size in mm
    ff          : fat fraction map in percent, same grid as mask (optional)
    kidney_mask : bool array of the kidney parenchyma on the same side (optional)
    """
    binary = np.asarray(mask) > 0.5
    n_vox = int(binary.sum())
    if n_vox == 0:
        return {}

    voxel_vol = float(np.prod(spacing))          # mm^3
    volume_ml = n_vox * voxel_vol / 1000.0
    out = {}

    def add(category, name, value, desc, unit):
        out[f'{roi}-{category}-{name}'] = [float(value), f'{roi} {desc}', unit, 'float']

    # --- Geometry ------------------------------------------------------
    labels, n_comp = ndi.label(binary, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labels.ravel())[1:]
    add('rsf_geom', 'number_of_components', n_comp, 'number of components', 'count')
    add('rsf_geom', 'largest_component_fraction', 100.0 * sizes.max() / n_vox,
        'largest component fraction', '%')

    # Local thickness proxy: 2 x distance to nearest background (zero-padded so
    # that masks touching the array edge are not overestimated)
    dist = ndi.distance_transform_edt(np.pad(binary, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
    thick = 2.0 * dist[binary]
    add('rsf_geom', 'mean_thickness', thick.mean() / 10.0, 'mean thickness', 'cm')

    # --- Fat fraction --------------------------------------------------
    if ff is not None:
        vals = np.asarray(ff, dtype=float)[binary]
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            add('rsf_ff', 'mean', vals.mean(), 'mean fat fraction', '%')
            add('rsf_ff', 'median', np.median(vals), 'median fat fraction', '%')
            add('rsf_ff', 'std', vals.std(ddof=1) if vals.size > 1 else np.nan,
                'fat fraction standard deviation', '%')
            add('rsf_ff', 'p10', np.percentile(vals, 10), 'fat fraction 10th percentile', '%')
            add('rsf_ff', 'p90', np.percentile(vals, 90), 'fat fraction 90th percentile', '%')
            add('rsf_ff', 'skewness', stats.skew(vals), 'fat fraction skewness', '')
            # Fat-equivalent volume: volume weighted by fat fraction
            add('rsf_ff', 'fat_equivalent_volume', vals.sum() / 100.0 * voxel_vol / 1000.0,
                'fat equivalent volume', 'mL')

            # Mean FF in the core (1 voxel eroded): less partial-volume contamination
            core = ndi.binary_erosion(binary)
            core_vals = np.asarray(ff, dtype=float)[core]
            core_vals = core_vals[np.isfinite(core_vals)]
            add('rsf_ff', 'mean_core',
                core_vals.mean() if core_vals.size else np.nan,
                'mean fat fraction (1-voxel eroded mask)', '%')
        else:
            logging.warning(f"{roi}: no finite fat fraction values inside the mask")

    # --- Relation to the kidney ---------------------------------------
    if kidney_mask is not None:
        kidney_ml = float(np.count_nonzero(kidney_mask)) * voxel_vol / 1000.0
        if kidney_ml > 0:
            add('rsf_kidney', 'kidney_volume', kidney_ml, 'kidney parenchyma volume', 'mL')
            add('rsf_kidney', 'sinus_to_kidney_volume_ratio', 100.0 * volume_ml / kidney_ml,
                'sinus fat to kidney parenchyma volume ratio', '%')
        else:
            logging.warning(f"{roi}: kidney mask empty, no ratio computed")

    return out