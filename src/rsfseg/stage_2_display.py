import os
import logging

import numpy as np
from tqdm import tqdm
import dbdicom as db
import pyvista as pv
from miblab import pipe

from rsfseg.utils.total_segmentator_class_maps import class_map
from rsfseg.utils import data
from miblab_plot import mosaic_overlay



PIPELINE = 'rsfseg'


def run(build, logfile, organs=None):

    task='total_mr'
    organs=['aorta']
    
    datapath = os.path.join(build, 'dixon', 'stage_5_clean_dixon_data')
    maskpath = os.path.join(build, 'rsfseg', 'stage_1_segment')
    displaypath = os.path.join(build, 'rsfseg', 'stage_2_display')

    # Controls
    group = "Controls"
    sitedatapath = os.path.join(datapath, group) 
    sitemaskpath = os.path.join(maskpath, group)
    sitedisplaypath = os.path.join(displaypath, group)

    run_site(sitedatapath, sitemaskpath, sitedisplaypath, organs, task=task)

    group = "Patients"
    for site in ['Exeter', 'Bari', 'Leeds', 'Bordeaux', 'Turku', 'Sheffield']:
        sitedatapath = os.path.join(datapath, group, site) 
        sitemaskpath = os.path.join(maskpath, group, site)
        sitedisplaypath = os.path.join(displaypath, group, site)

        run_site(sitedatapath, sitemaskpath, sitedisplaypath, organs, task=task)


def run_site(sitedatapath, sitemaskpath, sitedisplaypath, organs=None, task='total_mr'):
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

        # Skip if not the right task
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

        # # Skip if not in the right site
        # if site is not None:
        #     if patient_id[:4] not in SITE_IDS[site]:
        #         continue

        # Skip if file already exists
        png_file_orig = os.path.join(sitedisplaypath, f'{patient_id}_{study}_{sequence}')
        if os.path.exists(f"{png_file_orig}_{1}.png"):
             continue
        
        op_arr_orig = db.volume(series_op).values
        mask_arr_orig = db.volume(mask).values

        # Create images
        cnt = 0
        for transp in [(0,1,2), (0,2,1), (2,1,0)]:
        #for transp in ['coronal', 'axial', 'sagittal']:
            cnt += 1
            png_file = f"{png_file_orig}_{cnt}.png"
            op_arr = op_arr_orig.transpose(transp)
            mask_arr = mask_arr_orig.transpose(transp)
            # op_arr = op_arr_orig.reslice(orient=transp).values
            # mask_arr = mask_arr_orig.reslice(orient=transp).values

            rois = {}
            for idx, roi in class_map[task].items():
                rois[roi] = (mask_arr==idx).astype(np.int16)

            # Build mosaic
            if organs is None:
                mosaic_overlay(op_arr, rois, png_file, margin=[15,5,2])
            else:
                rois_k = {k:v for k, v in rois.items() if k in organs}
                if rois_k == {}:
                    raise ValueError(f'No organs {organs} found in {patient_id} {study}.')
                mosaic_overlay(op_arr, rois_k, png_file, margin=[15,5,2])


if __name__=='__main__':

    # Call like this to do all organs
    # python src/ibeat_rsfseg/stage_2_display.py --build=C:\Users\md1spsx\Documents\Data\iBEAt_Build


    # Call like this to do specific organs
    # python src/ibeat_rsfseg/stage_2_display.py --build=C:\Users\md1spsx\Documents\Data\iBEAt_Build --organs aorta liver

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



