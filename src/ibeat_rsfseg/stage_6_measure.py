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

from ibeat_rsfseg.utils import radiomics
from ibeat_rsfseg.utils.total_segmentator_class_maps import class_map

PIPELINE = 'rsfseg'

def run(build, logfile, organs=None):
    maskpath = os.path.join(build, 'rsfseg', 'stage_4_edit')
    measurepath = os.path.join(build, 'rsfseg', 'stage_6_measure', 'source_data')

    organs_to_measure = [f.name for f in os.scandir(maskpath) if f.is_dir()]
    if organs:
        organs_to_measure = [organ for organ in organs_to_measure if organ in organs]

    for organ in organs_to_measure:

        group = 'Controls'
        sitemaskpath = os.path.join(maskpath, organ, group)
        sitemeasurepath = os.path.join(measurepath, group) 
        measure_organ(sitemaskpath, sitemeasurepath, organ)

        group = 'Patients'   
        for site in ['Bari', 'Bordeaux', 'Exeter', 'Leeds', 'Sheffield', 'Turku']:
            sitemaskpath = os.path.join(maskpath, organ, group, site)
            sitemeasurepath = os.path.join(measurepath, group, site)
            measure_organ(sitemaskpath, sitemeasurepath, organ)
    
    concatenate(measurepath)   


def measure_organ(sitemaskpath, sitemeasurepath, organ):
    os.makedirs(sitemeasurepath, exist_ok=True)
    masks = db.series(sitemaskpath)

    #tasks = [measure_image(mask, sitemeasurepath, organ) for mask in masks]
    tasks = [delayed(measure_image)(mask, sitemeasurepath, organ) for mask in masks]

    Parallel(n_jobs=-1)(tasks)


def measure_image(automask, sitemeasurepath, organ):

    patient, study, series = automask[1], automask[2][0], automask[3][0]

    # If the results already exist, skip
    dmr_file = os.path.join(sitemeasurepath, f'{patient}_{study}_{series}')
    if os.path.exists(f'{dmr_file}.dmr.zip'):
        return
    
    print(f"Computing {dmr_file}")

    # Get mask volume 
    vol = db.volume(automask, verbose=0)

    # Init results
    dmr = {'data':{}, 'pars':{}}

    # Binary mask
    mask = (vol.values != 0).astype(np.float32)
    if np.sum(mask) == 0:
        return

    roi_vol = vreg.volume(mask, vol.affine)
    
    # Get skimage features
    try:
        results = radiomics.volume_features(roi_vol, organ)
    except Exception as e:
        logging.error(f"Patient {patient} {organ} - error computing ski-shapes: {e}")
    else:
        dmr['data'] = dmr['data'] | {p: v[1:] for p, v in results.items()}
        dmr['pars'] = dmr['pars'] | {(patient, study, p): v[0] for p, v in results.items()}

    # Get numpyradiomics shape features
    try:
        results = radiomics.shape_features_nprad(roi_vol, organ)
    except Exception as e:
        logging.error(f"Patient {patient} {organ} - error computing radiomics-shapes: {e}")
    else:
        dmr['data'] = dmr['data'] | {p:v[1:] for p, v in results.items()}
        dmr['pars'] = dmr['pars'] | {(patient, study, p): v[0] for p, v in results.items()}

    # Append parsed biomarkers in the dictionary for convenience
    dmr['columns'] = ['body_part', 'biomarker_category', 'biomarker']
    for p in dmr['data']:
        dmr['data'][p] += p.split('-')

    # Write results to file
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

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    kwargs = {
        "organs": {
            'type': str, 
            'default': None, 
            'nargs': '+',  # multiple arguments allowed separated by space
            'help': 'Organs',
        }
    }
    pipe.run_stage(run, BUILD, PIPELINE, __file__, **kwargs)