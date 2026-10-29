import os
import logging

import numpy as np
import cv2
import skimage
import scipy.ndimage as ndi
import matplotlib
matplotlib.use('Agg')  # geen scherm nodig, alleen wegschrijven
import matplotlib.pyplot as plt


# Mapping from anatomical plane name to the array axis that is iterated
# over 2D-slice-by-slice for that plane. axis=2 (axial) matches the
# original behaviour; axis=0 is coronal, axis=1 is sagittal.
_AXIS_BY_PLANE = {'axial': 2, 'coronal': 0, 'sagittal': 1}


def renal_sinus_fat(fat, water, kidneys, fixed_threshold=0.5, bounded=False,
                     max_dilation=8, pole_cut=False, pole_cut_planes=('coronal'),
                     min_notch_depth=3, margin=10, median_radius=1,
                     apply_median_filter=True, save_mosaic=None):
    """Compute renal sinus fat mask via convex hull of the kidney mask.

    fat: numpy array, fat-channel Dixon volume
    water: numpy array, water-channel Dixon volume, zelfde shape als
        `fat`. Wordt samen met `fat` gebruikt om de fat fraction map te
        berekenen (fat / (fat + water)); de fat/non-fat classificatie
        gebeurt op die fractiemap, niet op de ruwe fat-channel.
    kidneys: numpy array, kidney label mask (1=left, 2=right kidney)
    fixed_threshold: float, drempel op de fat fraction map: voxels met
        fat_fraction > fixed_threshold worden als vet beschouwd (geen
        Otsu/median_otsu meer).
    bounded: bool, if True the convex hull is additionally restricted to
        stay within `max_dilation` pixels of the kidney mask, to avoid
        leakage into pararenal fat through open sides of the sinus
    max_dilation: int, dilation radius (in pixels) used when bounded=True
    pole_cut: bool, if True the convex hull is additionally cut, per 2D
        slice, by a straight line connecting the two "poles" of the renal
        hilum notch (the two contour points bounding the deepest
        convexity defect on that slice). Only the hull area on the
        concave/kidney side of that line is kept, which trims hull area
        that bulges out past the natural hilum opening due to
        irregular/noisy contours.
    pole_cut_planes: tuple of str, which anatomical plane(s) to apply the
        pole cut in when pole_cut=True. Any of 'axial', 'coronal',
        'sagittal'. Defaults to ('axial',), matching the original
        behaviour. When multiple planes are given, the cut is computed
        independently in each plane (against the same starting hull) and
        the results are intersected, so a voxel is only kept if it
        survives the cut in every requested plane.
    min_notch_depth: float, minimum defect depth (in pixels) for a
        concavity to be treated as a real hilum notch when pole_cut=True.
    margin: int, padding (in voxels, in-plane) added around each kidney
        hull's bounding box before thresholding the fat fraction map.
    median_radius: int, only used if `apply_median_filter=True`: radius
        van de `scipy.ndimage` mediaanfilter die vóór het drempelen op
        de fat-fraction crop wordt toegepast.
    apply_median_filter: bool, if True (default), a
        `2*median_radius+1`-sized in-plane median filter is applied to
        the fat-fraction crop before thresholding, for noise
        suppression. Set to False to threshold the raw fat-fraction
        values directly.
    save_mosaic: str or None, if given, path (without extension) to save a
        per-kidney QC mosaic PNG showing mask / hull / pole line / fat /
        final mask for a handful of representative slices. One mosaic is
        saved per plane in `pole_cut_planes` (axial is always the
        original single-file behaviour; extra planes get a
        `_<plane>` suffix).
    """
    rsf = np.zeros(kidneys.shape)

    # Fat fraction map: alle thresholding gebeurt hierop, in plaats van
    # op de ruwe fat-channel.
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

        # Fat/non-fat classification: vaste drempel op de fat-fraction
        # map, per kidney, binnen een crop rond de convex hull.
        fat_mask = _fixed_threshold_mask(
            fat_fraction, kidney_hull, threshold=fixed_threshold,
            margin=margin, median_radius=median_radius,
            apply_median_filter=apply_median_filter)

        pole_lines = None
        if pole_cut:
            kidney_hull, pole_lines = _pole_restricted_hull_multi(
                mask, kidney_hull, min_notch_depth=min_notch_depth,
                planes=pole_cut_planes)

        if bounded:
            kidney_hull = _bounded_hull(mask, kidney_hull, max_dilation=max_dilation)

        sinus_fat = fat_mask * kidney_hull

        if not sinus_fat.any():
            if save_mosaic:
                logging.info(f"No sinus fat found for {side} kidney, skipping mosaic")
            continue

        sinus_fat_largest = _extract_significant_clusters_3d(sinus_fat)
        rsf[sinus_fat_largest] = kidney

        if save_mosaic:
            planes_to_show = pole_cut_planes if (pole_cut and pole_lines) else ('coronal',)
            for plane in planes_to_show:
                axis = _AXIS_BY_PLANE[plane]
                suffix = '' if plane == 'axial' else f'_{plane}'
                plane_pole_lines = (pole_lines or {}).get(plane, {})
                _save_qc_mosaic(
                    f"{save_mosaic}_{side}{suffix}.png", mask, kidney_hull, fat_mask,
                    sinus_fat, sinus_fat_largest, pole_lines=plane_pole_lines,
                    axis=axis)

    return rsf


def _fat_fraction_map(fat, water, eps=1e-6):
    """Fat fraction map: fat / (fat + water), geclipt naar [0, 1].

    eps voorkomt division-by-zero in achtergrondvoxels waar zowel fat
    als water ~0 zijn.
    """
    fat = fat.astype(float)
    water = water.astype(float)
    denom = fat + water + eps
    return np.clip(fat / denom, 0.0, 1.0)


def _fixed_threshold_mask(fat_fraction, hull, threshold=0.5, margin=10,
                           median_radius=1, apply_median_filter=True):
    """Vaste-drempel fat mask op de fractiemap: voxels met
    fat_fraction > threshold worden als vet beschouwd. Geen median_otsu.

    Beperkt tot de bounding box rond de kidney hull (+margin), puur om
    het rekenwerk te beperken -- de drempel zelf is niet
    data-afhankelijk, dus een crop is functioneel niet nodig, maar dit
    houdt gedrag/timing consistent met de lokale Otsu-variant.

    Als `apply_median_filter=True` wordt vóór het drempelen een
    2*median_radius+1-grote 2D-mediaanfilter (per slice, in-plane) op de
    crop toegepast, voor ruisonderdrukking vergelijkbaar met wat
    `median_otsu` intern doet. Zet op False om de rauwe fractiewaarden
    direct te drempelen.
    """
    fat_mask = np.zeros(fat_fraction.shape, dtype=float)
    bbox = _bbox_from_mask(hull, margin=margin)
    if bbox is None:
        return fat_mask

    ys, xs, zs = bbox
    region = fat_fraction[ys, xs, zs]

    if apply_median_filter and median_radius > 0:
        size = 2 * median_radius + 1
        filtered = np.empty_like(region)
        for i in range(region.shape[2]):
            filtered[:, :, i] = ndi.median_filter(region[:, :, i], size=size)
        region = filtered

    fat_mask[ys, xs, zs] = (region > threshold).astype(float)
    return fat_mask


def _bbox_from_mask(mask, margin=10):
    """Bounding box (tuple van slices) van `mask`, met `margin` voxels
    extra aan elke kant (geclipt aan de array-grenzen). None als leeg."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return tuple(
        slice(max(0, mn - margin), min(dim, mx + margin + 1))
        for mn, mx, dim in zip(mins, maxs, mask.shape)
    )


def _convex_hull_image_3d(array, **kwargs):

    volume = np.around(array)
    hull = np.zeros_like(volume, dtype=bool)
    for z in range(volume.shape[2]):
        slice_ = volume[:, :, z]
        if not slice_.any():
            continue
        hull[:, :, z] = skimage.morphology.convex_hull_image(slice_, **kwargs)
    return hull


def _bounded_hull(mask, hull, max_dilation=8):
    """Limit the convex hull to the region within `max_dilation` pixels
    of the kidney mask itself, avoiding leakage into pararenal fat
    through open sides of the sinus."""
    dilated_kidney = ndi.binary_dilation(mask, iterations=max_dilation)
    return hull & dilated_kidney


def _pole_restricted_hull_3d(mask, hull, min_notch_depth=3, axis=2):
    """Apply `_restrict_hull_by_pole_line` slice-by-slice over a 3D volume,
    slicing along `axis` (2=axial, 0=coronal, 1=sagittal).

    Returns the restricted hull plus a dict {index: (p_start, p_end, p_far)}
    of the detected pole-line points per slice along `axis` (only for
    slices where a notch was found), for QC/visualization purposes.
    """
    # Bring the requested axis to the end so the existing 2D slicing logic
    # (written for axis=2) works unchanged for any axis.
    mask_m = np.moveaxis(mask, axis, -1)
    hull_m = np.moveaxis(hull, axis, -1)
    restricted_m = np.zeros_like(hull_m, dtype=bool)
    pole_lines = {}
    for i in range(mask_m.shape[-1]):
        mask_slice = mask_m[:, :, i]
        hull_slice = hull_m[:, :, i]
        if not mask_slice.any() or not hull_slice.any():
            continue
        restricted_slice, pts = _restrict_hull_by_pole_line(
            mask_slice, hull_slice, min_notch_depth=min_notch_depth)
        restricted_m[:, :, i] = restricted_slice
        if pts is not None:
            pole_lines[i] = pts
    restricted = np.moveaxis(restricted_m, -1, axis)
    return restricted, pole_lines


def _pole_restricted_hull_multi(mask, hull, min_notch_depth=3, planes=('axial',)):
    """Apply the pole-cut restriction independently in one or more
    anatomical planes and intersect the results.

    Each plane's cut is computed against the same starting `hull` (not
    against the result of a previous plane), so the combination is
    order-independent: a voxel survives only if it is kept by the cut in
    every requested plane.

    Returns the combined restricted hull plus a dict
    {plane_name: {index: (p_start, p_end, p_far)}} of the pole-line points
    found in each plane, for QC/visualization purposes.
    """
    restricted = hull.copy()
    pole_lines = {}
    for plane in planes:
        if plane not in _AXIS_BY_PLANE:
            raise ValueError(
                f"Unknown pole_cut plane '{plane}', expected one of {list(_AXIS_BY_PLANE)}")
        axis = _AXIS_BY_PLANE[plane]
        plane_restricted, plane_pts = _pole_restricted_hull_3d(
            mask, hull, min_notch_depth=min_notch_depth, axis=axis)
        restricted = restricted & plane_restricted
        pole_lines[plane] = plane_pts
    return restricted, pole_lines


def _restrict_hull_by_pole_line(mask_slice, hull_slice, min_notch_depth=3):
    """Cut a 2D convex hull slice with a straight line connecting the two
    "poles" of the renal hilum notch: the two contour points bounding the
    deepest convexity defect of the kidney mask on this slice. Only the
    hull area on the concave (kidney) side of that line is kept.

    Falls back to returning the hull unchanged (pts=None) if no contour,
    no defects, or only shallow (noise-level) defects are found.
    """
    mask_u8 = (mask_slice > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return hull_slice, None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return hull_slice, None

    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 4:
        return hull_slice, None
    hull_idx = np.sort(hull_idx, axis=0)  # required to be monotonic by cv2

    try:
        defects = cv2.convexityDefects(contour, hull_idx)
    except cv2.error:
        return hull_slice, None
    if defects is None or len(defects) == 0:
        return hull_slice, None

    # Sommige OpenCV-builds/versies geven hier een (N, 4)-array terug in
    # plaats van de gedocumenteerde (N, 1, 4). Normaliseer dat hier zodat
    # de indexing hieronder altijd klopt.
    defects = np.asarray(defects)
    if defects.ndim == 2:
        defects = defects[:, np.newaxis, :]

    deepest = defects[np.argmax(defects[:, 0, 3])]
    s, e, f, depth = deepest[0]
    if depth / 256.0 < min_notch_depth:
        return hull_slice, None

    p_start = contour[s][0]  # (x, y)
    p_end = contour[e][0]
    p_far = contour[f][0]   # point inside the notch -> defines which side to keep

    yy, xx = np.mgrid[0:hull_slice.shape[0], 0:hull_slice.shape[1]]
    dx, dy = p_end[0] - p_start[0], p_end[1] - p_start[1]
    side = dx * (yy - p_start[1]) - dy * (xx - p_start[0])
    far_side = dx * (p_far[1] - p_start[1]) - dy * (p_far[0] - p_start[0])
    keep_mask = (side * far_side) >= 0

    return hull_slice & keep_mask, (p_start, p_end, p_far)


def _extract_largest_cluster_3d(array):
    structure = np.ones((3, 3, 3))  # 26-connectiviteit i.p.v. default 6
    label_img, cnt = ndi.label(array, structure=structure)
    sizes = ndi.sum(array, label_img, index=range(1, cnt + 1))
    max_label = np.argmax(sizes) + 1
    return label_img == max_label

def _extract_significant_clusters_3d(array, min_size_fraction=0, min_size_voxels=2):
    """Behoud alle 3D-samenhangende componenten boven een minimumgrootte,
    in plaats van alleen de allergrootste. Renaal sinusvet zit vaak
    opgesplitst in meerdere aparte pockets door de doorkruisende
    nierarterie/-vene/ureter; alleen de grootste component bewaren gooit
    echt sinusvet weg dat toevallig niet de grootste pocket is.

    min_size_fraction: componenten kleiner dan deze fractie van de
        grootste component worden weggegooid.
    min_size_voxels: absolute ondergrens (in voxels) om losse
        ruis-pixels alsnog te filteren, ongeacht de fractie hierboven.
    """
    structure = np.ones((3, 3, 3))
    label_img, cnt = ndi.label(array, structure=structure)
    if cnt == 0:
        return np.zeros_like(array, dtype=bool)
    sizes = ndi.sum(array, label_img, index=range(1, cnt + 1))
    max_size = sizes.max()
    threshold = max(min_size_voxels, min_size_fraction * max_size)
    keep_labels = [i + 1 for i, s in enumerate(sizes) if s >= threshold]
    return np.isin(label_img, keep_labels)

def _save_qc_mosaic(out_path, mask, hull, fat_mask, sinus_fat, sinus_fat_largest,
                     pole_lines=None, max_slices=6, axis=2):
    """Save a QC mosaic PNG comparing kidney mask, (restricted) hull, fat
    mask, hull ∩ fat and the final largest-cluster mask across a handful
    of representative slices, sliced along `axis` (2=axial, 0=coronal,
    1=sagittal). If `pole_lines` is given, the detected pole line for
    that plane is overlaid on the hull row.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pole_lines = pole_lines or {}

    # Bring the requested axis to the end so the rest of this function
    # (written for axis=2) works unchanged for any plane.
    mask, hull, fat_mask, sinus_fat, sinus_fat_largest = (
        np.moveaxis(v, axis, -1)
        for v in (mask, hull, fat_mask, sinus_fat, sinus_fat_largest)
    )

    volumes = [mask, hull, fat_mask, sinus_fat, sinus_fat_largest]
    titles = ['kidney mask', 'hull (± pole cut)', 'fat mask', 'hull ∩ fat', 'largest cluster']

    for vol, title in zip(volumes, titles):
        logging.info(f"{os.path.basename(out_path)} - {title}: nonzero={np.count_nonzero(vol)}")

    nonzero_slices = [z for z in range(mask.shape[2]) if mask[:, :, z].any()]
    if not nonzero_slices:
        nonzero_slices = list(range(mask.shape[2]))

    step = max(1, len(nonzero_slices) // max_slices)
    slices = nonzero_slices[::step][:max_slices]

    n_rows = len(volumes)
    n_cols = len(slices)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, (vol, title) in enumerate(zip(volumes, titles)):
        for col, z in enumerate(slices):
            ax = axes[row, col]
            ax.imshow(np.rot90(vol[:, :, z]), cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(title, fontsize=9)
            if row == 0:
                ax.set_title(f'z={z}', fontsize=9)
            if title.startswith('hull') and z in pole_lines:
                p_start, p_end, p_far = pole_lines[z]
                # np.rot90 rotates the displayed image; rotate the points to match
                h, w = vol.shape[0], vol.shape[1]
                def rot(p):
                    x, y = p
                    return (y, h - 1 - x)
                rs, re, rf = rot(p_start), rot(p_end), rot(p_far)
                ax.plot([rs[0], re[0]], [rs[1], re[1]], 'r-', linewidth=1.5)
                ax.plot(rf[0], rf[1], 'y*', markersize=6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)