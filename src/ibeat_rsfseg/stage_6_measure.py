import os
import json
import time
import logging
from pathlib import Path
from joblib import Parallel, delayed

import numpy as np
import dbdicom as db
import vreg
import pydmr
from miblab import pipe

from utils import radiomics

PIPELINE = 'rsfseg'

RSF_CLASS_MAP = {
    1: 'left_kidney_sinus_fat',
    2: 'right_kidney_sinus_fat',
}

# Stage 4 writes edited masks to
#   stage_4_edit/<Controls | Patients/<site>>/<task>/
# as series named '<task>_edited'. Both the current task and the earlier
# 'rsf_masks' task exist on disk; remove one here to measure only the other.
TASKS = ['renal_sinus_fat', 'rsf_masks']

SITES = ['Bari', 'Bordeaux', 'Exeter', 'Leeds', 'Sheffield', 'Turku']

# True: move the existing index.json aside (as index.json.bak) and let
# dbdicom rebuild it from the files on disk on every run.
REBUILD_INDEX = True

# Retries (and wait in s) when index.json is temporarily locked
RETRIES = 6
RETRY_DELAY = 2.0


def _retry(fn, what, attempts=RETRIES, delay=RETRY_DELAY):
    """Run fn; retry when a file is temporarily locked."""
    for i in range(1, attempts + 1):
        try:
            return fn()
        except (OSError, json.JSONDecodeError) as e:
            if i == attempts:
                raise
            logging.warning(
                f"{what}: attempt {i}/{attempts} failed ({e}); "
                f"retrying in {delay * i:.0f}s"
            )
            time.sleep(delay * i)


def run(build, logfile):
    maskpath = os.path.join(build, 'rsfseg', 'stage_4_edit')
    measurepath = os.path.join(build, 'rsfseg', 'stage_6_measure')

    for task in TASKS:
        taskmeasure = os.path.join(measurepath, task)

        measure_task(task,
                     os.path.join(maskpath, 'Controls', task),
                     os.path.join(taskmeasure, 'Controls'))

        for site in SITES:
            measure_task(task,
                         os.path.join(maskpath, 'Patients', site, task),
                         os.path.join(taskmeasure, 'Patients', site))

        concatenate(task, taskmeasure)


def _jobs(dbd, masks, sitemeasurepath):
    """Read volumes one at a time in the main process (no concurrent
    access to index.json) and hand the arrays to the workers."""
    for mask in masks:
        patient, study, series = mask[1], mask[2][0], mask[3][0]
        dmr_file = os.path.join(sitemeasurepath, f'{patient}_{study}_{series}')
        if os.path.exists(f'{dmr_file}.dmr.zip'):
            continue
        try:
            vol = _retry(lambda: dbd.volume(mask, None, 0),
                         f"reading {patient} {study}")
        except Exception as e:
            logging.error(f"{patient} {study}: error reading mask: {e}")
            continue
        labels = np.round(vol.values).astype(np.int16)
        yield delayed(measure_image)(patient, study, labels, vol.affine, dmr_file)


def measure_task(task, sitemaskpath, sitemeasurepath):
    if not os.path.isdir(sitemaskpath):
        logging.warning(f"{sitemaskpath}: no edited masks found, skipping.")
        return
    series_name = f'{task}_edited'
    os.makedirs(sitemeasurepath, exist_ok=True)

    # Force a fresh index: move the old index.json aside (kept as .bak) so
    # dbdicom has to rebuild it by scanning the files in the folder.
    if REBUILD_INDEX:
        index_file = os.path.join(sitemaskpath, 'index.json')
        if os.path.exists(index_file):
            _retry(lambda: os.replace(index_file, index_file + '.bak'),
                   f"moving old index.json {sitemaskpath}")
            logging.info(f"{sitemaskpath}: old index.json moved to index.json.bak")

    # Open the database once; volumes are read one at a time (see _jobs).
    dbd = _retry(lambda: db.open(sitemaskpath), f"opening {sitemaskpath}")
    try:
        all_series = _retry(dbd.series, f"listing {sitemaskpath}")
        masks = [m for m in all_series if m[3][0] == series_name]
        logging.info(f"{sitemaskpath}: {len(masks)} edited masks to measure")
        if not masks:
            logging.warning(
                f"{sitemaskpath}: no '{series_name}' series found. "
                f"{len(all_series)} series in this database, names: "
                f"{sorted({s[3][0] for s in all_series})[:10]}"
            )
        else:
            n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
            Parallel(n_jobs=n_jobs)(_jobs(dbd, masks, sitemeasurepath))
    finally:
        # Single writer, after all reading is done: saves the rebuilt
        # index.json, also when there were no masks or an error occurred.
        _retry(dbd.close, f"writing index.json {sitemaskpath}")


def measure_image(patient, study, labels, affine, dmr_file):
    dmr = {'data': {}, 'pars': {}}

    for idx, roi in RSF_CLASS_MAP.items():
        mask = (labels == idx).astype(np.float32)
        if mask.sum() == 0:
            continue

        roi_vol = vreg.volume(mask, affine)

        try:
            results = radiomics.volume_features(roi_vol, roi)
        except Exception as e:
            logging.error(f"Patient {patient} {roi} - error computing ski-shapes: {e}")
        else:
            dmr['data'] |= {p: v[1:] for p, v in results.items()}
            dmr['pars'] |= {(patient, study, p): v[0] for p, v in results.items()}

        try:
            results = radiomics.shape_features_nprad(roi_vol, roi)
        except Exception as e:
            logging.error(f"Patient {patient} {roi} - error computing radiomics-shapes: {e}")
        else:
            dmr['data'] |= {p: v[1:] for p, v in results.items()}
            dmr['pars'] |= {(patient, study, p): v[0] for p, v in results.items()}

    if not dmr['data']:
        logging.warning(f"{patient} {study}: no results, nothing written.")
        return

    dmr['columns'] = ['body_part', 'biomarker_category', 'biomarker']
    for p in dmr['data']:
        dmr['data'][p] += p.split('-')

    pydmr.write(dmr_file, dmr)


def concatenate(task, taskmeasure):
    outdir = Path(taskmeasure).parent.parent   # .../rsfseg

    for group in ['Controls', 'Patients']:
        folder = Path(os.path.join(taskmeasure, group))
        dmr_files = list(folder.rglob("*.dmr.zip"))
        if dmr_files == []:
            continue
        dmr_files = [str(f) for f in dmr_files]
        dmr_file = os.path.join(outdir, f'{task}_{group}_all_results.dmr.zip')
        pydmr.concat(dmr_files, dmr_file)

        # Create some derived formats for convenience

        # 1. Long format with additional columns (units, type, description)
        long_format_file = os.path.join(outdir, f'{task}_{group}_all_results_long.csv')
        pydmr.pars_to_long(dmr_file, long_format_file)

        # 2. Wide format
        wide_format_file = os.path.join(outdir, f'{task}_{group}_all_results_wide.csv')
        pydmr.pars_to_wide(dmr_file, wide_format_file)


if __name__ == '__main__':

    BUILD = r"X:\abdominal_imaging\Shared\Benthe\iBEAt_Build"

    pipe.run_stage(run, BUILD, PIPELINE, __file__)