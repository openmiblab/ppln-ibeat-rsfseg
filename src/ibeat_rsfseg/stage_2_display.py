import os
import logging

import numpy as np
from tqdm import tqdm
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

# All three orientations, indexed 1, 2, 3 (matches the _1.png, _2.png, _3.png
# filenames you already have). Set VIEWS_TO_SHOW below to pick a subset.
ALL_VIEWS = {
    1: (0, 1, 2),
    2: (0, 2, 1),
    3: (2, 1, 0),
}

# Which view(s) to generate. View 1 = coronal.
VIEWS_TO_SHOW = [1]


def run(build, logfile, organs=None):

    mask_task = 'rsf_masks'   

    datapath = os.path.join(build, 'dixon', 'stage_5_clean_dixon_data')
    maskpath = os.path.join(build, 'rsfseg_testing', 'stage_1_segment')
    displaypath = os.path.join(build, 'rsfseg_testing', 'stage_2_display')

    # Controls
    group = "Controls"
    sitedatapath = os.path.join(datapath, group)
    sitemaskpath = os.path.join(maskpath, group)
    sitedisplaypath = os.path.join(displaypath, group)

    run_site(sitedatapath, sitemaskpath, sitedisplaypath, organs, task=mask_task)

    group = "Patients"
    for site in ['Exeter', 'Bari', 'Leeds', 'Bordeaux', 'Turku', 'Sheffield']:
        sitedatapath = os.path.join(datapath, group, site)
        sitemaskpath = os.path.join(maskpath, group, site)
        sitedisplaypath = os.path.join(displaypath, group, site)

        run_site(sitedatapath, sitemaskpath, sitedisplaypath, organs, task=mask_task)


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


def run_site(sitedatapath, sitemaskpath, sitedisplaypath, organs=None, task='rsf_masks'):
    # Build output folders
    if organs is None:
        sitedisplaypath = os.path.join(sitedisplaypath, f'mosaic_{task}')
    else:
        sitedisplaypath = os.path.join(sitedisplaypath, 'mosaic_' + '_'.join(organs))
    os.makedirs(sitedisplaypath, exist_ok=True)

    record = data.dixon_record()
    all_series = db.series(sitedatapath)

    # Loop over the masks
    for mask in tqdm(db.series(sitemaskpath), 'Displaying masks..'):

        # Skip if not the right task (i.e. not an rsf_masks series)
        if mask[3][0] != task:
            continue

        # Get the outphase series for the mask
        patient_id = mask[1]
        study = mask[2][0]
        sequence = data.dixon_series_desc(record, patient_id, study)
        series_op = [sitedatapath, patient_id, mask[2], (f'{sequence}_out_phase', 0)]

        # Skip if Dixon series is not there
        if series_op not in all_series:
            continue

        # Skip if file already exists (checks the first view we will generate)
        png_file_orig = os.path.join(sitedisplaypath, f'{patient_id}_{study}_{sequence}')
        first_view = VIEWS_TO_SHOW[0]
        if os.path.exists(f"{png_file_orig}_{first_view}.png"):
            continue
        try:
            op_arr_orig = db.volume(series_op).values
            mask_arr_orig = db.volume(mask).values
        except Exception as e:
            logging.error(f"Error reading data for {patient_id} {sequence}: {e}")
            continue

        # Round mask values before comparing: after a DICOM write/read round-trip,
        # values that should be exactly 1 or 2 can come back as e.g. 0.9998 or 2.0003
        # due to rescale slope/intercept quantization, which makes an exact `==`
        # comparison silently fail and produce an empty ROI.
        mask_arr_orig = np.round(mask_arr_orig).astype(np.int16)

        # Skip entirely if there is no mask at all for this patient/study
        if not mask_arr_orig.any():
            logging.info(f"No nonzero mask values for {patient_id} {study}, skipping.")
            continue

        # Create images, only for the selected view(s)
        for cnt in VIEWS_TO_SHOW:
            transp = ALL_VIEWS[cnt]
            png_file = f"{png_file_orig}_{cnt}.png"
            op_arr = op_arr_orig.transpose(transp)
            mask_arr = mask_arr_orig.transpose(transp)

            # Only keep the slices (along the last axis, i.e. axis=2 after
            # transposing) where the kidney/RSF mask is actually present,
            # with a small padding margin for context.
            slice_range = _mask_slice_range(mask_arr, axis=2, pad=2)
            if slice_range is None:
                # No mask visible in this view at all, skip this orientation
                continue
            op_arr_cropped = op_arr[:, :, slice_range]
            mask_arr_cropped = mask_arr[:, :, slice_range]

            rois = {}
            for idx, roi in RSF_CLASS_MAP.items():
                rois[roi] = (mask_arr_cropped == idx).astype(np.int16)

            # Build mosaic
            if organs is None:
                mosaic_overlay(op_arr_cropped, rois, png_file, margin=[15, 5, 2])
            else:
                rois_k = {k: v for k, v in rois.items() if k in organs}
                if rois_k == {}:
                    raise ValueError(f'No organs {organs} found in {patient_id} {study}.')
                mosaic_overlay(op_arr_cropped, rois_k, png_file, margin=[15, 5, 2])


if __name__ == '__main__':

    # Call like this to do both kidneys
    # python src/rsfseg/stage_2_display.py --build=C:\Users\...\iBEAt_Build

    # Call like this for one kidney specifically
    # python src/rsfseg/stage_2_display.py --build=C:\Users\...\iBEAt_Build --organs left_kidney_sinus_fat

    BUILD = r"X:\abdominal_imaging\Shared\Benthe"
    kwargs = {
        "organs": {
            "type": str,
            "default": None,
            "nargs": "+",
            "help": "Organs (left_kidney_sinus_fat / right_kidney_sinus_fat)",
        }
    }
    pipe.run_stage(run, BUILD, PIPELINE, __file__, **kwargs)