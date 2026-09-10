import os
import logging

from tqdm import tqdm
import dbdicom as db

from utils import data
from utils import rsf_original as rsf


def segment(datapath, kidneypath, maskpath, group, site=None, patient=None, study=None,
            max_patients=None):

    if site is None:
        sitedatapath = os.path.join(datapath, group)
        sitekidneypath = os.path.join(kidneypath, group)
        sitemaskpath = os.path.join(maskpath, group)
    else:
        sitedatapath = os.path.join(datapath, group, site)
        sitekidneypath = os.path.join(kidneypath, group, site)
        sitemaskpath = os.path.join(maskpath, group, site)
    os.makedirs(sitemaskpath, exist_ok=True)

    record = data.dixon_record()

    series = db.series(sitedatapath)
    series_out_phase = [s for s in series if s[3][0][-9:] == 'out_phase']
    all_kidney_masks = db.series(sitekidneypath)

    processed_patients = set()

    for series_op in tqdm(series_out_phase, desc='Computing RSF..'):

        patient_op = series_op[1]
        study_op = series_op[2][0]
        series_op_desc = series_op[3][0]
        sequence = series_op_desc[:-10]

        if patient is not None and patient != patient_op:
            continue
        if study is not None and study != study_op:
            continue

        selected_sequence = data.dixon_series_desc(record, patient_op, study_op)
        if sequence != selected_sequence:
            continue

        # Skip if the fat series doesn't exist
        series_fat = series_op[:3] + [(sequence + '_fat', 0)]
        if series_fat not in series:
            continue

        # Skip if the kidney mask doesn't exist
        kidney_series = [sitekidneypath, patient_op, (study_op, 0), ('kidney_masks', 0)]
        if kidney_series not in all_kidney_masks:
            continue

        # Skip if the RSF mask already exists
        mask_study = [sitemaskpath, patient_op, (study_op, 0)]
        mask_series = mask_study + [('rsf_masks', 0)]
        if mask_series in db.series(mask_study):
            processed_patients.add(patient_op)
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
        processed_patients.add(patient_op)


if __name__ == '__main__':

    DATAPATH = r"X:\abdominal_imaging\Shared\Benthe\dixon\stage_5_clean_dixon_data"
    KIDNEYPATH = r"X:\abdominal_imaging\Shared\Benthe\kidneyvol\stage_3_edit"
    BUILDPATH = r"X:\abdominal_imaging\Shared\Benthe\rsfseg\stage_1_segment"
    os.makedirs(BUILDPATH, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(BUILDPATH, 'error.log')),
            logging.StreamHandler(),  # ook naar de console, zodat silent failures zichtbaar worden
        ]
    )

    #segment(DATAPATH, KIDNEYPATH, BUILDPATH, 'Controls')
    for site in ['Bari', 'Bordeaux', 'Exeter', 'Leeds', 'Sheffield', 'Turku']:
        segment(DATAPATH, KIDNEYPATH, BUILDPATH, 'Patients', site=site)