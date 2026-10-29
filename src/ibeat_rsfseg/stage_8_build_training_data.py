"""Build nnU-Net training data from the edited RSF masks.

Channels: out_phase, in_phase, water, fat, fat fraction (noNorm).
Labels: edited masks from stage_4_edit/stage_1_segment (left=1, right=2).
"""

import os
import json
import logging

import numpy as np
from tqdm import tqdm
import dbdicom as db
from miblab import pipe

from utils import data


PIPELINE = 'rsfseg'

# Mask series (edited in stage 1 for now, changing later to 4) to use as training labels.
TASKS = ['rsf_masks']

DATASET = 'Dataset001_iBEAtRSF'

SITES = ['Exeter', 'Bari', 'Leeds', 'Bordeaux', 'Turku', 'Sheffield']


def run(build, logfile):

    datapath = os.path.join(build, 'dixon', 'stage_5_clean_dixon_data')
    editpath = os.path.join(build, 'rsfseg', 'stage_1_segment')
    trainpath = os.path.join(build, 'rsfseg', 'stage_8_build_training_data')
    scratchpath = os.path.join(build, 'rsfseg', 'stage_8_scratch')  # series used for nifti export

    database = os.path.join(trainpath, DATASET)
    os.makedirs(os.path.join(database, 'imagesTr'), exist_ok=True)
    os.makedirs(os.path.join(database, 'labelsTr'), exist_ok=True)

    # Controls
    group = "Controls"
    for task in TASKS:
        run_site(
            os.path.join(datapath, group),
           #os.path.join(editpath, group, task),
            os.path.join(editpath, group),
            database, os.path.join(scratchpath, group), task,
        )

    group = "Patients"
    for site in SITES:
        for task in TASKS:
            run_site(
                os.path.join(datapath, group, site),
               # os.path.join(editpath, group, site, task),
                os.path.join(editpath, group, site),
                database, os.path.join(scratchpath, group, site), task,
            )

    build_json(database)


def run_site(sitedatapath, siteeditpath, database, sitescratchpath, task):

    images_tr = os.path.join(database, 'imagesTr')
    labels_tr = os.path.join(database, 'labelsTr')
    os.makedirs(sitescratchpath, exist_ok=True)

    record = data.dixon_record()
    all_dixon_series = db.series(sitedatapath)

    # Loop over the edited masks
    for edited_series in tqdm(db.series(siteeditpath), desc=f'Building {siteeditpath[-30:]}..'):

        # Skip if not the correct task
        if edited_series[3][0] != task: # f"{task}_edited":
            continue

        patient = edited_series[1]
        study = edited_series[2][0]
        case_id = f"{patient}_{study}".replace(' ', '_')
        prefix = os.path.join(images_tr, case_id)
        niis = [f"{prefix}_000{i}.nii.gz" for i in range(5)]  # op, ip, water, fat, ff
        nii_label = os.path.join(labels_tr, f"{case_id}.nii.gz")

        # Skip if the case is complete (label is written last)
        if all(os.path.exists(f) for f in niis + [nii_label]):
            continue

        # Get the Dixon series
        sequence = data.dixon_series_desc(record, patient, study)
        study_dixon = [sitedatapath, patient, edited_series[2]]
        series_op = study_dixon + [(f'{sequence}_out_phase', 0)]
        series_ip = study_dixon + [(f'{sequence}_in_phase', 0)]
        series_wi = study_dixon + [(f'{sequence}_water', 0)]
        series_fi = study_dixon + [(f'{sequence}_fat', 0)]
        dixon_series = [series_op, series_ip, series_wi, series_fi]

        # Skip if any of the Dixon series is missing
        if not all(s in all_dixon_series for s in dixon_series):
            logging.info(f"{patient} {study}: Dixon series incomplete, skipping.")
            continue

        try:
            fat = db.volume(series_fi, verbose=0)
            water = db.volume(series_wi, verbose=0)
            label = db.volume(edited_series, verbose=0)

            # Fat fraction = fat / (fat + water), in [0, 1]
            den = fat.values + water.values
            ff = np.divide(fat.values, den, out=np.zeros(den.shape, dtype=float), where=den > 1e-6)
            ff = np.clip(ff, 0, 1).astype(np.float32)

            # Round labels: DICOM round-trip can give e.g. 0.9998 or 2.0003
            label_arr = np.round(label.values).astype(np.int16)
        except Exception as e:
            logging.exception(f"{case_id}: error reading data: {e}")
            continue

        try:
            # Write FF and clean label to scratch, so that all channels
            # go through the same db.to_nifti
            scratch_study = [sitescratchpath, patient, (study, 0)]
            series_ff = scratch_study + [(f'{sequence}_fat_fraction', 0)]
            series_lbl = scratch_study + [('rsf_label', 0)]
            db.write_volume((ff, fat.affine), series_ff, ref=series_fi, verbose=0)
            db.write_volume((label_arr, label.affine), series_lbl, ref=series_op, verbose=0)

            for s, f in zip([series_op, series_ip, series_wi, series_fi, series_ff], niis):
                db.to_nifti(s, f, verbose=0)
            db.to_nifti(series_lbl, nii_label, verbose=0)  # label last
            logging.info(f"{case_id}: successfully written training data")
        except Exception as e:
            logging.exception(f"{case_id}: error writing training data: {e}")


def build_json(database):

    labels_tr = os.path.join(database, 'labelsTr')
    num_training = len([f for f in os.listdir(labels_tr) if f.endswith('.nii.gz')])

    json_data = {
        "channel_names": {
            "0": "out_phase",
            "1": "in_phase",
            "2": "water",
            "3": "fat",
            "4": "noNorm"       # fat fraction, 0-1
        },
        "labels": {
            "background": 0,
            "renal_sinus_fat_left": 1,
            "renal_sinus_fat_right": 2
        },
        "numTraining": num_training,
        "file_ending": ".nii.gz"
    }
    with open(os.path.join(database, 'dataset.json'), 'w') as f:
        json.dump(json_data, f, indent=2)


if __name__ == '__main__':

    BUILD = r"X:\abdominal_imaging\Shared\Benthe\iBEAt_Build"
    pipe.run_stage(run, BUILD, PIPELINE, __file__)