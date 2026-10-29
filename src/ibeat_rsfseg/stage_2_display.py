import os
import logging

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import dbdicom as db
import pyvista as pv
from miblab import pipe

from utils import data
from miblab_plot import mosaic_overlay


PIPELINE = 'rsfseg'

RSF_CLASS_MAP = {
    1: 'left_kidney_sinus_fat',
    2: 'right_kidney_sinus_fat',
}

PARENCHYMA_TASK = 'kidney_masks'
PARENCHYMA_CLASS_MAP = {
    1: 'left_kidney_parenchyma',
    2: 'right_kidney_parenchyma',
}

ALL_VIEWS = {
    1: (0, 1, 2),
    2: (0, 2, 1),
    3: (2, 1, 0),
}

# Which view(s) to generate. View 1 = coronal.
VIEWS_TO_SHOW = [1]

# Which image to use as the mosaic background. One of:
#   'fat_fraction', 'fat', 'water', 'in_phase', 'out_phase'
BACKGROUND = 'fat_fraction'

BACKGROUND_SUFFIXES = {
    'fat': ['_fat'],
    'water': ['_water'],
    'in_phase': ['_in_phase'],
    'out_phase': ['_out_phase'],
    'fat_fraction': ['_fat', '_water'],
}

# Small constant to avoid division by zero when computing the fat fraction.
FF_EPSILON = 1e-6

N_JOBS = -1


def run(build, logfile):

    mask_task = 'rsf_masks'

    datapath = os.path.join(build, 'dixon', 'stage_5_clean_dixon_data')
    maskpath = os.path.join(build, 'rsfseg', 'stage_1_segment')
    kidneyvolpath = os.path.join(build, 'kidneyvol', 'stage_3_edit')
    displaypath = os.path.join(build, 'rsfseg', 'stage_2_display')

    # Controls
    group = "Controls"
    sitedatapath = os.path.join(datapath, group)
    sitemaskpath = os.path.join(maskpath, group)
    sitekidneyvolpath = os.path.join(kidneyvolpath, group)
    sitedisplaypath = os.path.join(displaypath, group)

    run_site(sitedatapath, sitemaskpath, sitekidneyvolpath, sitedisplaypath, task=mask_task)

    group = "Patients"
    for site in ['Exeter', 'Bari', 'Leeds', 'Bordeaux', 'Turku', 'Sheffield']:
        sitedatapath = os.path.join(datapath, group, site)
        sitemaskpath = os.path.join(maskpath, group, site)
        sitekidneyvolpath = os.path.join(kidneyvolpath, group, site)
        sitedisplaypath = os.path.join(displaypath, group, site)

        run_site(sitedatapath, sitemaskpath, sitekidneyvolpath, sitedisplaypath, task=mask_task)


def _mask_slice_range(mask_arr, axis=2, pad=2):
    """Return a slice object covering only the range along `axis` where
    the mask has nonzero values, padded by `pad` slices on each side
    (clipped to the array bounds). Returns None if the mask is empty."""
    other_axes = tuple(a for a in range(mask_arr.ndim) if a != axis)
    present = np.any(mask_arr != 0, axis=other_axes)
    idx = np.where(present)[0]
    if idx.size == 0:
        return None
    lo = max(0, idx.min() - pad)
    hi = min(mask_arr.shape[axis] - 1, idx.max() + pad)
    return slice(lo, hi + 1)


def _find_parenchyma_mask(all_parenchyma_series, patient_id, study):
    """Find the parenchyma (kidneyvol) mask series matching the given
    patient and study, filtered by PARENCHYMA_TASK. Returns None if not found."""
    for series in all_parenchyma_series:
        if series[1] != patient_id:
            continue
        if series[2] != study:
            continue
        if series[3][0] != PARENCHYMA_TASK:
            continue
        return series
    return None


def _fat_fraction(fat_arr, water_arr):
    """Compute the fat fraction map (in %) from fat and water images.

    FF = fat / (fat + water) * 100
    """
    fat_arr = fat_arr.astype(np.float32)
    water_arr = water_arr.astype(np.float32)
    denom = fat_arr + water_arr
    ff = np.where(denom > FF_EPSILON, fat_arr / (denom + FF_EPSILON), 0.0)
    ff = np.clip(ff, 0.0, 1.0) * 100.0
    return ff


def _build_background(arrays, background):
    """Given a dict {suffix: array} holding the raw volumes needed for the
    chosen background, return the background array to display."""
    if background == 'fat_fraction':
        return _fat_fraction(arrays['_fat'], arrays['_water'])
    suffix = BACKGROUND_SUFFIXES[background][0]
    return arrays[suffix].astype(np.float32)


def _process_one_mask(mask, sitedatapath, all_series, all_parenchyma_series,
                       sitedisplaypath, task, organs, background, needed_suffixes):
    """Process a single mask series: build the mosaic overlay PNG(s) for it.

    """
    # Skip if not the right task (i.e. not an rsf_masks series)
    if mask[3][0] != task:
        return None

    patient_id = mask[1]
    study = mask[2][0]

    try:
        record = data.dixon_record()
        sequence = data.dixon_series_desc(record, patient_id, study)
    except Exception as e:
        return f"ERROR: could not resolve sequence name for {patient_id} {study}: {e}"

    # Build the list of Dixon series needed for the chosen background
    needed_series = {
        suffix: [sitedatapath, patient_id, mask[2], (f'{sequence}{suffix}', 0)]
        for suffix in needed_suffixes
    }

    # Skip if any required Dixon series is not there
    if any(series not in all_series for series in needed_series.values()):
        return None

    # Skip if file already exists (checks the first view we will generate)
    png_file_orig = os.path.join(sitedisplaypath, f'{patient_id}_{study}_{sequence}')
    first_view = VIEWS_TO_SHOW[0]
    if os.path.exists(f"{png_file_orig}_{first_view}.png"):
        return None

    parenchyma_series = _find_parenchyma_mask(all_parenchyma_series, patient_id, mask[2])
    if parenchyma_series is None:
        return (f"WARNING: No parenchyma mask found for {patient_id} {study}, "
                f"skipping crop-by-parenchyma (check PARENCHYMA_TASK / kidneyvol path).")

    try:
        background_arrays = {
            suffix: db.volume(series).values
            for suffix, series in needed_series.items()
        }
        mask_arr_orig = db.volume(mask).values
        parenchyma_arr_orig = db.volume(parenchyma_series).values
    except Exception as e:
        return f"ERROR: reading data for {patient_id} {sequence}: {e}"

    mask_arr_orig = np.round(mask_arr_orig).astype(np.int16)
    parenchyma_arr_orig = np.round(parenchyma_arr_orig).astype(np.int16)

    if not mask_arr_orig.any():
        return f"INFO: No nonzero mask values for {patient_id} {study}, skipping."

    if not parenchyma_arr_orig.any():
        return f"INFO: No nonzero parenchyma mask values for {patient_id} {study}, skipping."

    bg_arr_orig = _build_background(background_arrays, background)

    for cnt in VIEWS_TO_SHOW:
        transp = ALL_VIEWS[cnt]
        png_file = f"{png_file_orig}_{cnt}.png"
        bg_arr = bg_arr_orig.transpose(transp)
        mask_arr = mask_arr_orig.transpose(transp)
        parenchyma_arr = parenchyma_arr_orig.transpose(transp)

        slice_range = _mask_slice_range(parenchyma_arr, axis=2, pad=2)
        if slice_range is None:
            continue
        bg_arr_cropped = bg_arr[:, :, slice_range]
        mask_arr_cropped = mask_arr[:, :, slice_range]

        rois = {}
        for idx, roi in RSF_CLASS_MAP.items():
            rois[roi] = (mask_arr_cropped == idx).astype(np.int16)

        if background == 'fat_fraction':
            vmin, vmax = 0, 100
        else:
            vmin, vmax = None, None

        # Build mosaic
        if organs is None:
            mosaic_overlay(bg_arr_cropped, rois, png_file, margin=[15, 5, 2],
                            vmin=vmin, vmax=vmax)
        else:
            rois_k = {k: v for k, v in rois.items() if k in organs}
            if rois_k == {}:
                return f"ERROR: No organs {organs} found in {patient_id} {study}."
            mosaic_overlay(bg_arr_cropped, rois_k, png_file, margin=[15, 5, 2],
                            vmin=vmin, vmax=vmax)

    return None


def run_site(sitedatapath, sitemaskpath, sitekidneyvolpath, sitedisplaypath,
             organs=None, task='rsf_masks', background=None, n_jobs=None):
    if background is None:
        background = BACKGROUND
    if background not in BACKGROUND_SUFFIXES:
        raise ValueError(
            f"Unknown background '{background}', choose one of {list(BACKGROUND_SUFFIXES)}."
        )
    needed_suffixes = BACKGROUND_SUFFIXES[background]

    if n_jobs is None:
        n_jobs = N_JOBS

    # Build output folders
    if organs is None:
        sitedisplaypath = os.path.join(sitedisplaypath, f'mosaic_{background}_{task}')
    else:
        sitedisplaypath = os.path.join(sitedisplaypath, 'mosaic_' + '_'.join(organs))
    os.makedirs(sitedisplaypath, exist_ok=True)

    all_series = db.series(sitedatapath)
    all_parenchyma_series = db.series(sitekidneyvolpath)
    masks = list(db.series(sitemaskpath))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_one_mask)(
            mask, sitedatapath, all_series, all_parenchyma_series,
            sitedisplaypath, task, organs, background, needed_suffixes,
        )
        for mask in tqdm(masks, desc='Displaying masks..')
    )

    for message in results:
        if message is None:
            continue
        if message.startswith('ERROR'):
            logging.error(message)
        elif message.startswith('WARNING'):
            logging.warning(message)
        else:
            logging.info(message)


if __name__ == '__main__':

    build = r"X:\abdominal_imaging\Shared\Benthe"
    pipe.run_stage(run, build, PIPELINE, __file__)