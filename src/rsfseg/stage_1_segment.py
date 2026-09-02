import os
import logging

from tqdm import tqdm
import dbdicom as db
import miblab
import torch

from utils import data



def segment(datapath, maskpath, group, site=None, task='total_mr', patient=None, study=None):

    if site is None:
        sitedatapath = os.path.join(datapath, group)
        sitemaskpath = os.path.join(maskpath, group)
    else:
        sitedatapath = os.path.join(datapath, group, site)
        sitemaskpath = os.path.join(maskpath, group, site)
    os.makedirs(sitemaskpath, exist_ok=True)

    # List of selected dixon series
    record = data.dixon_record()

    # Get out phase series
    series = db.series(sitedatapath)
    series_out_phase = [s for s in series if s[3][0][-9:]=='out_phase']

    # Loop over the out-phase series
    for series_op in tqdm(series_out_phase, desc='Segmenting..'):

        # Patient and output study
        patient_op = series_op[1]
        study_op = series_op[2][0]
        series_op_desc = series_op[3][0]
        sequence = series_op_desc[:-10]

        # Skip if not the right patient or study
        if patient is not None:
            if patient != patient_op:
                continue
        if study is not None:
            if study != study_op:
                continue

        # Skip if it is not the right sequence
        selected_sequence = data.dixon_series_desc(record, patient_op, study_op)
        if sequence != selected_sequence:
            continue

        # Skip if the masks already exist
        mask_study = [sitemaskpath, patient_op, (study_op,0)]
        mask_series = mask_study + [(f'{task}', 0)]
        if mask_series in db.series(mask_study):
            continue

        # Other source data series
        series_wi = series_op[:3] + [(sequence + '_water', 0)]

        # Read volumes
        if series_wi in db.series(series_op[:3]):
            vol = db.volume(series_wi, verbose=0)
        else:
            vol = db.volume(series_op, verbose=0)

        # Perform segmentation
        try:
            device = 'gpu' if torch.cuda.is_available() else 'cpu'
            label_vol = miblab.rsfseg(vol, cutoff=0.01, task=task, device=device)
        except Exception as e:
            logging.error(f"Error processing {patient_op} {sequence} with total segmentator: {e}")
            continue

        # Save results
        db.write_volume(label_vol, mask_series, ref=series_op, verbose=0)
            

if __name__=='__main__':

    DATAPATH = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build\dixon\stage_2_data"
    BUILDPATH = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build\rsfseg\stage_1_segment"
    os.makedirs(BUILDPATH, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(BUILDPATH, 'error.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    segment(DATAPATH, BUILDPATH, 'Controls')
    # segment(DATAPATH, BUILDPATH, 'Patients', site='Sheffield')