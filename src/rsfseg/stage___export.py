import os
import argparse
from datetime import date

import pandas as pd
import pydmr



def export_to_redcap(measurepath, resultspath):
    """
    Create export for upload to redcap repository
    """
    today = date.today().strftime("%Y-%m-%d")
    os.makedirs(resultspath, exist_ok=True)

    def visit_nr(value):
        if value == 'Baseline':
            return 0
        if value == 'Followup':
            return 2
        if value[:5] == 'Visit':
            return int(value[5]) - 1
        
    def fix_exeter_volunteer(harmonized_id, visit_nr):
        # Correct a mistake in ID 
        # Exeter volunteer 3 is the same person as volunteer 1
        # This needs to be removed when the issue is fixed at the source
        if harmonized_id == 'iBE-3128C03':
            harmonized_id = 'iBE-3128C01'
            visit_nr += 2
        return harmonized_id, visit_nr

    for group in ['Controls', 'Patients']:
        dmr_file = os.path.join(measurepath, f'{group}_rsfseg_auto')
        dmr = pydmr.read(dmr_file)

        # Append parsed biomarkers in the dictionary
        dmr_output_file = os.path.join(resultspath, f'{group}_OrgansShape_{today}')
        dmr['columns'] = ['body_part', 'image', 'biomarker_category', 'biomarker']
        for p in dmr['data']:
            dmr['data'][p] = dmr['data'][p][:4] + ['mask'] + dmr['data'][p][4:]

        # Change PatientIDs to central format
        pars_harmonized = {}
        for p,v in dmr['pars'].items():
            harmonized_id = f"iBE-{p[0].replace('_','')}"
            visit = visit_nr(p[1])
            harmonized_id, visit = fix_exeter_volunteer(harmonized_id, visit)
            pars_harmonized[(harmonized_id, visit, p[2])] = v
        dmr['pars'] = pars_harmonized
        pydmr.write(dmr_output_file, dmr)

        # Drop partically covered ROIs
        to_drop = [
            'clavicula_left', 
            'clavicula_right',
            'femur_left',
            'femur_right',
            'humerus_left',
            'humerus_right',
            'prostate',
            'scapula_left',
            'scapula_right',
            'urinary_bladder',
            'gallbladder',
            'heart',
            'lung_right',
            'lung_left',
            'gluteus_maximus_right',
            'gluteus_maximus_left',
            'gluteus_medius_left',
            'gluteus_medius_right',
            'gluteus_minimus_left',
            'gluteus_minimus_right',
            'sacrum',
            'hip_left',
            'hip_right',
            'kidney_left', # edited kidneys delivered under kidneyvol
            'kidney_right',
        ]
        pydmr.drop(dmr_output_file, body_part=to_drop)

        # 1. Save in long format with additional columns
        long_format_file = os.path.join(resultspath, f'{group}_OrgansShape_{today}.csv')
        pydmr.pars_to_long(dmr_output_file, long_format_file)
        # Replace column names

        # 2. Save in wide format
        wide_format_file = os.path.join(resultspath, f'{group}_OrgansShape_{today}_wide.csv')
        pydmr.pars_to_wide(dmr_output_file, wide_format_file)

        # Replace column names
        new_cols = {
            "subject": "harmonized_id",
            "study": "visit_nr",
            "value": "result",
        }
        df = pd.read_csv(long_format_file)
        df.rename(columns=new_cols, inplace=True)
        df.to_csv(long_format_file, index=False)
        df = pd.read_csv(wide_format_file)
        df.rename(columns=new_cols, inplace=True)
        df.to_csv(wide_format_file, index=False)


       


if __name__ == '__main__':

    PROJ_DIR=r"C:\Users\md1spsx\Documents\Data\iBEAt_Build\rsfseg"
    DATA_DIR=os.path.join(PROJ_DIR, "stage_3_measure")
    BUILD_DIR=os.path.join(PROJ_DIR, "stage_3_export")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=DATA_DIR, help="Data folder")
    parser.add_argument("--build", type=str, default=BUILD_DIR, help="Build folder")
    args = parser.parse_args()

    export_to_redcap(args.data, args.build)