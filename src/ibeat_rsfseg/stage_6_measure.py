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

from utils import data
from utils import radiomics
from utils import rsf_features

PIPELINE = 'rsfseg'

RSF_CLASS_MAP = {
    1: 'left_kidney_sinus_fat',
    2: 'right_kidney_sinus_fat',
}

KIDNEY_LABELS = {
    'left_kidney_sinus_fat': [1],
    'right_kidney_sinus_fat': [2],
}

TASKS = ['rsf_masks']
SITES = ['Bari', 'Bordeaux', 'Exeter', 'Leeds', 'Sheffield', 'Turku']

REBUILD_INDEX = False
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


def _key(s):
    """(patient, study name, series name) - ignores the index numbers."""
    return (str(s[1]), s[2][0], s[3][0])


def run(build, logfile):
    maskpath = os.path.join(build, 'rsfseg', 'stage_4_edit')
    measurepath = os.path.join(build, 'rsfseg', 'stage_6_measure')
    dixonpath = os.path.join(build, 'dixon', 'stage_5_clean_dixon_data')
    kidneypath = os.path.join(build, 'kidneyvol', 'stage_3_edit')

    for task in TASKS:
        taskmeasure = os.path.join(measurepath, task)

        measure_task(task,
                     os.path.join(maskpath, 'Controls', task),
                     os.path.join(taskmeasure, 'Controls'),
                     os.path.join(dixonpath, 'Controls'),
                     os.path.join(kidneypath, 'Controls'))

        for site in SITES:
            measure_task(task,
                         os.path.join(maskpath, 'Patients', site, task),
                         os.path.join(taskmeasure, 'Patients', site),
                         os.path.join(dixonpath, 'Patients', site),
                         os.path.join(kidneypath, 'Patients', site))

        concatenate(task, taskmeasure)


def _read_volume(dbd, series, what):
    vol = _retry(lambda: dbd.volume(series, None, 0), what)
    return vol.values, vol.affine


def _load_extras(patient, study, dixon_dbd, dixon_index, kidney_dbd, kidney_index, record):
    """Fat, water and kidney volumes for one patient/study (None when missing)."""
    extras = {'fat': None, 'water': None, 'kidney': None}

    try:
        sequence = data.dixon_series_desc(record, patient, study)
        fat_s = dixon_index.get((str(patient), study, f'{sequence}_fat'))
        water_s = dixon_index.get((str(patient), study, f'{sequence}_water'))
        if fat_s is None or water_s is None:
            logging.warning(f"{patient} {study}: fat/water series not found, no FF features.")
        else:
            extras['fat'] = _read_volume(dixon_dbd, fat_s, f"reading fat {patient} {study}")
            extras['water'] = _read_volume(dixon_dbd, water_s, f"reading water {patient} {study}")
    except Exception as e:
        logging.error(f"{patient} {study}: error reading fat/water: {e}")
        extras['fat'] = extras['water'] = None

    try:
        kidney_s = kidney_index.get((str(patient), study, 'kidney_masks'))
        if kidney_s is None:
            logging.warning(f"{patient} {study}: kidney mask not found, no kidney ratio.")
        else:
            extras['kidney'] = _read_volume(kidney_dbd, kidney_s, f"reading kidney {patient} {study}")
    except Exception as e:
        logging.error(f"{patient} {study}: error reading kidney mask: {e}")

    return extras


def _jobs(dbd, masks, sitemeasurepath, ctx):
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
        extras = _load_extras(patient, study, **ctx)
        yield delayed(measure_image)(patient, study, labels, vol.affine, dmr_file, extras)


def measure_task(task, sitemaskpath, sitemeasurepath, sitedixonpath, sitekidneypath):
    if not os.path.isdir(sitemaskpath):
        logging.warning(f"{sitemaskpath}: no edited masks found, skipping.")
        return
    series_name = f'{task}_edited'
    os.makedirs(sitemeasurepath, exist_ok=True)

    if REBUILD_INDEX:
        index_file = os.path.join(sitemaskpath, 'index.json')
        if os.path.exists(index_file):
            _retry(lambda: os.replace(index_file, index_file + '.bak'),
                   f"moving old index.json {sitemaskpath}")
            logging.info(f"{sitemaskpath}: old index.json moved to index.json.bak")

    ctx = {'record': data.dixon_record(),
           'dixon_dbd': None, 'dixon_index': {},
           'kidney_dbd': None, 'kidney_index': {}}
    try:
        ctx['dixon_dbd'] = _retry(lambda: db.open(sitedixonpath), f"opening {sitedixonpath}")
        ctx['dixon_index'] = {_key(s): s for s in ctx['dixon_dbd'].series()}
    except Exception as e:
        logging.error(f"{sitedixonpath}: cannot open dixon database ({e}); no FF features.")
    try:
        ctx['kidney_dbd'] = _retry(lambda: db.open(sitekidneypath), f"opening {sitekidneypath}")
        ctx['kidney_index'] = {_key(s): s for s in ctx['kidney_dbd'].series()}
    except Exception as e:
        logging.error(f"{sitekidneypath}: cannot open kidney database ({e}); no kidney ratio.")

    # Open the mask database once; volumes are read one at a time (see _jobs).
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
            Parallel(n_jobs=n_jobs)(_jobs(dbd, masks, sitemeasurepath, ctx))
    finally:

        _retry(dbd.close, f"writing index.json {sitemaskpath}")


def _collect(dmr, patient, study, results):
    dmr['data'] |= {p: v[1:] for p, v in results.items()}
    dmr['pars'] |= {(patient, study, p): v[0] for p, v in results.items()}


def measure_image(patient, study, labels, affine, dmr_file, extras=None):
    dmr = {'data': {}, 'pars': {}}
    extras = extras or {}


    ff = None
    kidney = None
    try:
        if extras.get('fat') is not None and extras.get('water') is not None:
            fat, fat_affine = extras['fat']
            water, water_affine = extras['water']
            rsf_features.check_same_grid(fat, fat_affine, affine, labels.shape, 'fat image')
            rsf_features.check_same_grid(water, water_affine, affine, labels.shape, 'water image')
            ff = rsf_features.fat_fraction_map(fat, water)
    except Exception as e:
        logging.error(f"{patient} {study}: fat fraction map not computed: {e}")
    try:
        if extras.get('kidney') is not None:
            kidney_arr, kidney_affine = extras['kidney']
            rsf_features.check_same_grid(kidney_arr, kidney_affine, affine, labels.shape, 'kidney mask')
            kidney = np.round(kidney_arr).astype(np.int16)
    except Exception as e:
        logging.error(f"{patient} {study}: kidney mask not used: {e}")

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
            _collect(dmr, patient, study, results)

        try:
            results = radiomics.shape_features_nprad(roi_vol, roi)
        except Exception as e:
            logging.error(f"Patient {patient} {roi} - error computing radiomics-shapes: {e}")
        else:
            _collect(dmr, patient, study, results)

        try:
            kidney_mask = None
            if kidney is not None:
                kidney_mask = np.isin(kidney, KIDNEY_LABELS[roi])
            results = rsf_features.rsf_features(
                roi, mask, roi_vol.spacing, ff=ff, kidney_mask=kidney_mask)
        except Exception as e:
            logging.error(f"Patient {patient} {roi} - error computing RSF features: {e}")
        else:
            _collect(dmr, patient, study, results)

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