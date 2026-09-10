import os
import logging

from tqdm import tqdm
import dbdicom as db
from miblab import pipe

from utils import data
from utils import rsf_original as rsf

PIPELINE = 'rsf'

def run(build, logfile):
    run_site(build, 'Controls')
    for site in ['Exeter', 'Leeds', 'Bari', 'Bordeaux', 'Sheffield', 'Turku']:
        run_site(build, 'Patients', site=site)


def run_site(build, group, site=None):

    # Define site paths
    if group == 'Controls':
        dixonpath = os.path.join(build, 'dixon', 'stage_5_clean_dixon_data', group) 
        kidneypath = os.path.join(build, 'kidneyvol', 'stage_3_edit', group)
        rsfpath = os.path.join(build, 'rsfseg', 'stage_1_segment', group)
    else:
        dixonpath = os.path.join(build, 'dixon', 'stage_5_clean_dixon_data', group, site)
        kidneypath = os.path.join(build, 'stage_4_compute_fatwater', group, site)
        rsfpath = os.path.join(build, 'rsfseg', 'stage_1_segment', group, site)
    os.makedirs(rsfpath, exist_ok=True)

    record = data.dixon_record()

    all_dixon_series = db.series(dixonpath)
    all_dixon_out_phase_series = [s for s in all_dixon_series if s[3][0][-9:] == 'out_phase']
    all_kidney_masks = db.series(kidneypath)

    for series_op in tqdm(all_dixon_out_phase_series, desc='Computing RSF..'):

        patient_op = series_op[1]
        study_op = series_op[2][0]
        series_op_desc = series_op[3][0]
        sequence = series_op_desc[:-10]

        selected_sequence = data.dixon_series_desc(record, patient_op, study_op)
        if sequence != selected_sequence:
            continue

        # Skip if the fat series doesn't exist
        series_fat = series_op[:3] + [(sequence + '_fat', 0)]
        if series_fat not in all_dixon_series:
            continue

        # Skip if the kidney mask doesn't exist
        kidney_series = [kidneypath, patient_op, (study_op, 0), ('kidney_masks', 0)]
        if kidney_series not in all_kidney_masks:
            continue

        # Skip if the RSF mask already exists
        mask_study = [rsfpath, patient_op, (study_op, 0)]
        mask_series = mask_study + [('rsf_masks', 0)]
        if mask_series in db.series(mask_study):
            continue

        # Read the data
        try:
            fat = db.volume(series_fat, verbose=0)
            kidney_label = db.volume(kidney_series, verbose=0)
        except Exception as e:
            logging.exception(f"Error reading data for {patient_op} {sequence}: {e}")
            continue

        # Perform convex hull RSF calculation
        try:
            rsf_values = rsf.renal_sinus_fat(fat.values, kidney_label.values)
        except Exception as e:
            logging.exception(f"Error computing RSF for {patient_op} {sequence}: {e}")
            continue

        # Save results
        db.write_volume((rsf_values, fat.affine), mask_series, ref=series_fat, verbose=0)

        # series = [path/to/database, patient_id, (study_description, 1), (series_description, 1)]
        # series = [/users/rsf, '001', ('rsf_masks', 0)]


if __name__ == '__main__':

    build = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"
    pipe.run_stage(run, build, PIPELINE, __file__)