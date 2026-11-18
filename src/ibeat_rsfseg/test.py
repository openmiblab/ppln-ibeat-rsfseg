"""Open napari for a single patient: out-phase image + automask.

Read-only: nothing is written to the edit database.

Usage:
    python open_napari_one.py --site Leeds
    python open_napari_one.py --site Leeds --patient 1234 --plane coronal
    python open_napari_one.py --site Leeds --screenshot docs/napari_edit.png
"""

import os
import argparse

import numpy as np
import napari
import dbdicom as db

from utils import data

BUILD = r"X:\abdominal_imaging\Shared\Benthe\iBEAt_Build"
TASK = 'rsf_masks'

TRANSP = {
    'coronal': (0, 1, 2),
    'sagittal': (2, 1, 0),
    'axial': (0, 2, 1),
}


def find_case(data_db, mask_db, patient=None):
    """Return (out-phase series, mask series) of the first matching case."""
    masks = {(str(s[1]), s[2][0]): s
             for s in mask_db.series() if s[3][0] == TASK}
    try:
        record = data.dixon_record()
    except FileNotFoundError as e:
        # The record (which dixon sequence was selected per study) is not
        # available here; fall back to the first out-phase series found.
        print(f"Warning: {e}\nSkipping the selected-sequence check.")
        record = None

    for s in data_db.series():
        if not s[3][0].endswith('out_phase') or s[3][1] != 0:
            continue
        if patient is not None and str(s[1]) != str(patient):
            continue
        mask_series = masks.get((str(s[1]), s[2][0]))
        if mask_series is None:
            continue
        sequence = s[3][0][:-len('_out_phase')]
        if record is not None and sequence != data.dixon_series_desc(record, s[1], s[2][0]):
            continue
        return s, mask_series

    raise SystemExit("No matching patient with an out-phase image and mask found.")


def main(site, patient=None, plane='coronal', build=BUILD, screenshot=None,
         group='Patients'):
    datapath = os.path.join(build, 'dixon', 'stage_5_clean_dixon_data', group, site)
    maskpath = os.path.join(build, 'rsfseg', 'stage_1_segment', group, site)

    data_db = db.open(datapath)
    mask_db = db.open(maskpath)

    series_op, series_mask = find_case(data_db, mask_db, patient)
    print(f"Opening patient {series_op[1]}, study {series_op[2][0]}")

    op = data_db.volume(series_op, None, 0)
    mask = mask_db.volume(series_mask, None, 0)

    transp = TRANSP[plane]
    image = op.values.transpose(transp)
    labels = np.round(mask.values).astype(np.int16).transpose(transp)

    # Generic layer names, so no patient identifiers end up in screenshots
    viewer = napari.Viewer()
    viewer.add_image(image, name='out-phase', colormap='gray')
    viewer.add_labels(labels, name='automask')

    if screenshot:
        os.makedirs(os.path.dirname(os.path.abspath(screenshot)), exist_ok=True)
        # Scroll to a good slice first if you like, then close the window.
        # The screenshot is taken when you press Enter in the terminal.
        input("Position the viewer, then press Enter to save the screenshot...")
        viewer.screenshot(screenshot, canvas_only=True)
        print(f"Saved screenshot to {screenshot}")

    napari.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', required=True)
    parser.add_argument('--patient', default=None)
    parser.add_argument('--plane', default='coronal', choices=list(TRANSP))
    parser.add_argument('--build', default=BUILD)
    parser.add_argument('--screenshot', default=None,
                        help='Optional path of a PNG to save from the viewer')
    parser.add_argument('--group', default='Patients')
    args = parser.parse_args()
    main(args.site, args.patient, args.plane, args.build, args.screenshot, args.group)