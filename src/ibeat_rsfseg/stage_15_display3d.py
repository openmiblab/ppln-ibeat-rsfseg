"""
Stage 15 - 3D display of RSF-maskers (reference vs predicted), per fold and per kidney.

Reads the output of stage 13:
    <build>/rsfseg/stage_13_test/fold_X/predictions/<case>.nii.gz
and the labels from stage 8:
    <build>/rsfseg/stage_8_build_training_data/Dataset001_iBEAtRSF/labelsTr/<case>.nii.gz

Output:
    <build>/rsfseg/stage_15_display3d/fold_X/left/    (mosaic-PNG's)
    <build>/rsfseg/stage_15_display3d/fold_X/right/

With --anonymize the case names (patient_study) are replaced by anonymous IDs
(S01, S02, ...) in the tile labels, and the output goes to a separate folder:
    <build>/rsfseg/stage_15_display3d_anonymized/...
The key that links IDs to case names is written OUTSIDE that folder, so the
output folder can be shared without the key:
    <build>/rsfseg/stage_15_anonymization_key.csv      (keep private!)
The IDs are the same in every fold and for both kidneys, and are reproducible
as long as the set of cases and --seed do not change.

Use:
    python stage_15_display3d.py
    python stage_15_display3d.py --folds 0 2 --sides left
    python stage_15_display3d.py --folds ensemble --gauss 1.5 --smooth 30
    python stage_15_display3d.py --gauss 0 --smooth 0        (no smoothing)
    python stage_15_display3d.py --anonymize                 (anonymized tile labels)
"""

import os
import csv
import glob
import random
import argparse

import numpy as np
import nibabel as nib

from miblab_plot import rotating_mosaics_da

STAGE = 'stage_15_display3d'
PIPELINE = 'rsfseg'
DATASET = 'Dataset001_iBEAtRSF'
SIDES = {'left': 1, 'right': 2}


class _LazyMask:
    """Loads a nifti mask `[:]` and selects a label.

    Has also a `spacing` attribute (from the header) for correct proportions.
    """

    def __init__(self, path, value):
        self.path = path
        self.value = value
        self.spacing = tuple(float(z) for z in nib.load(path).header.get_zooms()[:3])

    def __getitem__(self, key):
        data = np.rint(np.asanyarray(nib.load(self.path).dataobj))
        return (data == self.value)[key]


def make_anonymous_ids(labels_dir, seed):
    """Map every case in labelsTr to an anonymous ID (S01, S02, ...).

    The cases are shuffled with a fixed seed, so the IDs do not follow the
    alphabetical order of the patient names, and the mapping is reproducible.
    """
    cases = sorted(f[:-len('.nii.gz')] for f in os.listdir(labels_dir) if f.endswith('.nii.gz'))
    random.Random(seed).shuffle(cases)
    width = max(2, len(str(len(cases))))
    return {case: f'S{i + 1:0{width}d}' for i, case in enumerate(cases)}


def write_key(ids, path):
    """Write the anonymous ID -> case name key (keep this file private)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['anonymous_id', 'case'])
        for case, anon in sorted(ids.items(), key=lambda kv: kv[1]):
            writer.writerow([anon, case])


def run_fold(fold_dir, labels_dir, out_dir, subjects_per_chunk, nviews, columns, n,
             sides, gauss, smooth_iter, ids=None):
    pred_files = sorted(glob.glob(os.path.join(fold_dir, 'predictions', '*.nii.gz')))
    if n > 0:
        pred_files = pred_files[:n]
    if not pred_files:
        print(f'Geen predicties gevonden in {fold_dir}, overgeslagen')
        return

    # Anonymized: order the tiles by anonymous ID, not by patient name
    if ids is not None:
        pred_files.sort(key=lambda p: ids.get(os.path.basename(p)[:-len('.nii.gz')], 'zzz'))

    for side in sides:
        masks, labels = [], []
        for pred_file in pred_files:
            case = os.path.basename(pred_file)[:-len('.nii.gz')]
            label_file = os.path.join(labels_dir, f'{case}.nii.gz')
            if not os.path.isfile(label_file):
                print(f'Label ontbreekt voor {case}, overgeslagen')
                continue
            name = ids[case] if ids is not None else case
            masks.append(_LazyMask(label_file, SIDES[side]))
            labels.append(f'{name} {side} ref')
            masks.append(_LazyMask(pred_file, SIDES[side]))
            labels.append(f'{name} {side} pred')

        side_dir = os.path.join(out_dir, side)
        os.makedirs(side_dir, exist_ok=True)
        rotating_mosaics_da(
            dir_output=side_dir,
            masks=masks,
            labels=labels,
            chunksize=2 * subjects_per_chunk,  # even -> ref/pred blijven samen
            nviews=nviews,
            columns=columns,
            gauss=gauss,
            smooth_iter=smooth_iter,
        )
        print(f'{os.path.basename(fold_dir)} {side}: {len(masks)} tegels naar {side_dir}')


def run(build, folds=('0', '1', '2', '3', '4'), n=0, subjects_per_chunk=10,
        nviews=25, columns=4, sides=('left', 'right'), gauss=1.0, smooth_iter=20,
        anonymize=False, seed=42):
    if columns % 2 != 0:
        raise ValueError('columns moet even zijn zodat ref en pred naast elkaar staan.')

    build_rsf = os.path.join(build, PIPELINE)
    stage13 = os.path.join(build_rsf, 'stage_13_test')
    labels_dir = os.path.join(build_rsf, 'stage_8_build_training_data', DATASET, 'labelsTr')

    ids = None
    stage_name = STAGE
    if anonymize:
        ids = make_anonymous_ids(labels_dir, seed)
        stage_name = STAGE + '_anonymized'
        # The key is stored next to (not inside) the output folder
        key_file = os.path.join(build_rsf, 'stage_15_anonymization_key.csv')
        write_key(ids, key_file)
        print(f'Anonimiseringssleutel (privé houden!): {key_file}')
    output = os.path.join(build_rsf, stage_name)

    for fold in folds:
        name = 'ensemble' if fold == 'ensemble' else f'fold_{fold}'
        run_fold(os.path.join(stage13, name), labels_dir, os.path.join(output, name),
                 subjects_per_chunk, nviews, columns, n, sides, gauss, smooth_iter, ids=ids)


if __name__ == '__main__':
    BUILD = r"X:\abdominal_imaging\Shared\Benthe\iBEAt_Build"

    parser = argparse.ArgumentParser()
    parser.add_argument('--build', type=str, default=BUILD, help='Build folder')
    parser.add_argument('--folds', type=str, nargs='+', default=['0', '1', '2', '3', '4'],
                        help="Folds, bv. --folds 0 2, of 'ensemble'")
    parser.add_argument('--sides', type=str, nargs='+', default=['left', 'right'],
                        choices=['left', 'right'], help='Welke nier(en), telkens apart')
    parser.add_argument('--n', type=int, default=0, help='Max cases per fold (0 = alle)')
    parser.add_argument('--per-chunk', type=int, default=10, help='Subjects per mozaiek')
    parser.add_argument('--nviews', type=int, default=25)
    parser.add_argument('--columns', type=int, default=4)
    parser.add_argument('--gauss', type=float, default=1.0,
                        help='Gaussian sigma in voxels op het volume (0 = uit)')
    parser.add_argument('--smooth', type=int, default=20,
                        help='Taubin-iteraties op het oppervlak (0 = uit)')
    parser.add_argument('--anonymize', action='store_true',
                        help='Vervang casenamen op de tegels door anonieme ID\'s '
                             '(S01, S02, ...); output naar stage_15_display3d_anonymized')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed voor de volgorde van de anonieme ID\'s')
    args = parser.parse_args()

    run(args.build, folds=tuple(args.folds), n=args.n,
        subjects_per_chunk=args.per_chunk, nviews=args.nviews, columns=args.columns,
        sides=tuple(args.sides), gauss=args.gauss, smooth_iter=args.smooth,
        anonymize=args.anonymize, seed=args.seed)