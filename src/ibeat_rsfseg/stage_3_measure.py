import os
import logging
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

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

TASK = 'rsf_masks'


def run(build, logfile):
    maskpath = os.path.join(build, 'rsfseg', 'stage_1_segment')
    measurepath = os.path.join(build, 'rsfseg', 'stage_3_measure')

    measure_task(os.path.join(maskpath, 'Controls'),
                 os.path.join(measurepath, 'Controls'))

    for site in ['Bari', 'Bordeaux', 'Exeter', 'Leeds', 'Sheffield', 'Turku']:
        measure_task(os.path.join(maskpath, 'Patients', site),
                     os.path.join(measurepath, 'Patients', site))

    concatenate(measurepath)


def measure_task(sitemaskpath, sitemeasurepath):
    os.makedirs(sitemeasurepath, exist_ok=True)
    masks = db.series(sitemaskpath)   # bouwt/valideert de index
    masks = [m for m in masks if m[3][0] == TASK]

    n_jobs = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    Parallel(n_jobs=n_jobs)(
        delayed(measure_image)(mask, sitemeasurepath) for mask in masks
    )

def measure_image(automask, sitemeasurepath):
    patient, study, series = automask[1], automask[2][0], automask[3][0]

    if series != TASK:
        return

    dmr_file = os.path.join(sitemeasurepath, f'{patient}_{study}_{series}')
    if os.path.exists(f'{dmr_file}.dmr.zip'):
        return

    vol = db.volume(automask, verbose=0)
    labels = np.round(vol.values).astype(np.int16)

    dmr = {'data': {}, 'pars': {}}

    for idx, roi in RSF_CLASS_MAP.items():
        mask = (labels == idx).astype(np.float32)
        if mask.sum() == 0:
            continue

        roi_vol = vreg.volume(mask, vol.affine)

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
        logging.warning(f"{patient} {study}: geen resultaten, niets geschreven.")
        return

    dmr['columns'] = ['body_part', 'biomarker_category', 'biomarker']
    for p in dmr['data']:
        dmr['data'][p] += p.split('-')

    pydmr.write(dmr_file, dmr)


def concatenate(measurepath):

    for group in ['Controls', 'Patients']:
        folder = os.path.join(measurepath, group) 
        folder = Path(folder)
        dmr_files = list(folder.rglob("*.dmr.zip"))
        if dmr_files == []:
            continue
        dmr_files = [str(f) for f in dmr_files]
        dmr_file = os.path.join(Path(measurepath).parent, f'{group}_all_results.dmr.zip')
        pydmr.concat(dmr_files, dmr_file)

        # Create some derived formats for convenience

        # 1. Long format with additional columns (units, type, description)
        long_format_file = os.path.join(Path(measurepath).parent, f'{group}_all_results_long.csv')
        pydmr.pars_to_long(dmr_file, long_format_file)

        # 2. Wide format
        wide_format_file = os.path.join(Path(measurepath).parent, f'{group}_all_results_wide.csv')
        pydmr.pars_to_wide(dmr_file, wide_format_file)



if __name__=='__main__':

    # python src/ibeat_rsfseg/stage_3_measure.py --build=C:\Users\md1spsx\Documents\Data\iBEAt_Build --organs aorta

    BUILD = r"X:\abdominal_imaging\Shared\Benthe\iBEAt_Build"

    pipe.run_stage(run, BUILD, PIPELINE, __file__)