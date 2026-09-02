import os

import dbdicom as db
import vreg
from totalsegmentator.map_to_binary import class_map

from utils import render, edit, data

DIXONS = os.path.join(os.getcwd(), 'build', 'dixon', 'stage_2_data')
SEGMENTATIONS = os.path.join(os.getcwd(), 'build', 'rsfseg', 'stage_1_segment')
EDITED = os.path.join(os.getcwd(), 'build', 'rsfseg', 'stage_4_edit')


def display_surface():
    patient = '1128_C01'
    study = 'Visit1'
    classes = {
    #    "tissue_types_mr": ["torso_fat"], 
        "total_mr": ["liver","pancreas"],
    }
    opacity = [1, 1, 1, 1]
    masks = []
    for task in classes:
        series = [SEGMENTATIONS, patient, study, task]
        vol = db.volume(series)
        label_map = {v: k for k, v in class_map[task].items()}
        masks += [(vol.values == label_map[s]) for s in classes[task]]
    render.surface_with_clipping(masks, vol.spacing, above_opacity=0.2, n_iter=60, relaxation=0.2, opacity=opacity)


def display_editor():

    patient = '1128_C01'
    study = 'Visit1'
    organ = 'pancreas'
    
    # Get image
    record = data.dixon_record()
    sequence = data.dixon_series_desc(record, patient, study)
    series_op = [DIXONS, patient, study, sequence + '_out_phase']
    op = db.volume(series_op)

    # Show automask
    task = 'total_mr'
    auto_label_series = [SEGMENTATIONS, patient, study, task]
    auto_label = db.volume(auto_label_series)
    edit.edit_mask_with_napari(op.values, auto_label.values)

    # Show edited mask
    edited_mask_series = [EDITED, patient, (study, 0), (f"{organ}_edited", 0)]
    edit_label = db.volume(edited_mask_series)
    edit.edit_mask_with_napari(op.values, edit_label.values)


if __name__=='__main__':
    # display_surface()
    display_editor()
