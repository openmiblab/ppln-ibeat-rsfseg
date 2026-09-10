import os
import logging

import numpy as np
from tqdm import tqdm
import dbdicom as db
import pyvista as pv
from miblab import pipe

from ibeat_rsfseg.utils.total_segmentator_class_maps import class_map
from ibeat_rsfseg.utils import data
from miblab_plot import mosaic_overlay



PIPELINE = 'rsfseg'


def run(build, logfile, organs=None):
    datapath = os.path.join(build, 'dixon', 'stage_5_clean_dixon_data')
    maskpath = os.path.join(build, 'rsfseg', 'stage_4_edit')
    displaypath = os.path.join(build, 'rsfseg', 'stage_5_display')

    organs_to_measure = [f.name for f in os.scandir(maskpath) if f.is_dir()]
    if organs:
        organs_to_measure = [organ for organ in organs_to_measure if organ in organs]

    for organ in organs_to_measure:

        # Controls
        group = "Controls"
        sitedatapath = os.path.join(datapath, group) 
        sitemaskpath = os.path.join(maskpath, organ, group)
        sitedisplaypath = os.path.join(displaypath, organ, group)

        run_site(sitedatapath, sitemaskpath, sitedisplaypath, organ)

        group = "Patients"
        for site in ['Exeter', 'Bari', 'Leeds', 'Bordeaux', 'Turku', 'Sheffield']:
            sitedatapath = os.path.join(datapath, group, site) 
            sitemaskpath = os.path.join(maskpath, organ, group, site)
            sitedisplaypath = os.path.join(displaypath, organ, group, site)

            run_site(sitedatapath, sitemaskpath, sitedisplaypath, organ)


def run_site(sitedatapath, sitemaskpath, sitedisplaypath, organ):
    # Build output folders
    sitedisplaypath = os.path.join(sitedisplaypath, f'mosaic_{organ}')
    os.makedirs(sitedisplaypath, exist_ok=True)

    record = data.dixon_record()
    all_series = db.series(sitedatapath)

    # Loop over the masks
    for mask in tqdm(db.series(sitemaskpath), 'Displaying masks..'):

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
            rois[organ] = (mask_arr!=0).astype(np.int16)
            mosaic_overlay(op_arr, rois, png_file, margin=[15,5,2])


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



