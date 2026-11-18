import os
import logging
import argparse

from tqdm import tqdm
import dbdicom as db
import numpy as np
from miblab_dl import rsf

import utils.data


def run(build):
    for site in ['Leeds', 'Exeter', 'Bari', 'Bordeaux', 'Sheffield', 'Turku']:
        run_site(build, 'Patients', site=site)
    run_site(build, 'Controls')
    

def run_site(build, group, site=None):

    # Define database paths
    input = 'stage_5_clean_dixon_data'
    output = 'stage_14_rsf_segmentation'

    if group == 'Controls':
        db_dixon = os.path.join(build, 'dixon', input, group) 
        db_rsf = os.path.join(build, 'rsf', output, group)
    else:
        db_dixon = os.path.join(build, 'dixon', input, group, site)
        db_rsf = os.path.join(build, 'rsf', output, group, site)

    # List of selected dixon series
    record = utils.data.dixon_record()

    # Get all out_phase series
    dixon_series = db.series(db_dixon)
    all_op_series = [s for s in dixon_series if s[3][0][-9:]=='out_phase']

    # List existing series so they can be skipped in the loop
    existing_series = db.series(db_rsf)

    for series_op in tqdm(all_op_series, desc='Computing renal sinus fat'):

        try:

            # Identifiers
            patient = series_op[1]
            study = series_op[2][0]
            sequence = series_op[3][0][:-10] # remove "_out_phase"

            # Skip if it is not the right sequence
            selected_sequence = utils.data.dixon_series_desc(record, patient, study)
            if sequence != selected_sequence:
                continue

            # Skip if the results already exist
            series_rsf = [db_rsf, patient, (study, 0), (f'renal_sinus_fat', 0)]
            if series_rsf in existing_series:
                continue

            # Other source data series
            series_ip = series_op[:3] + [(sequence + '_in_phase', 0)]
            series_wi = series_op[:3] + [(sequence + '_water', 0)]
            series_fi = series_op[:3] + [(sequence + '_fat', 0)]

            # Read dixon data
            vol = [
                db.volume(series_op),
                db.volume(series_ip),
                db.volume(series_wi),
                db.volume(series_fi),
            ]

            # Predict rsf labels
            dixon = np.stack([v.values for v in vol], axis=-1)
            values_rsf = rsf(dixon)
            db.write_volume((values_rsf, vol[0].affine), series_rsf, ref=series_op)

            logging.info(
                f'Segmented renal sinus fat for patient {patient}, study {study}, sequence {sequence}'
            )

        except:

            logging.exception(
                f'Error segmenting patient {patient}, study {study}, sequence {sequence}'
            )

        

if __name__=='__main__':

    BUILD = r"X:\abdominal_imaging\Shared\Benthe\iBEAt_Build"
    os.makedirs(BUILD, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    rsf_build = os.path.join(args.build, 'rsf')
    os.makedirs(rsf_build, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(rsf_build, 'stage_9_rsf_segmentation.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    run(args.build)
            


