"""
Stage 15 - 3D display of RSF-maskers (reference vs predicted), per fold and per kidney.

Reads the output of stage 13:
    <build>/rsfseg/stage_13_test/fold_X/predictions/<case>.nii.gz
and the labels from stage 8:
    <build>/rsfseg/stage_8_build_training_data/Dataset001_iBEAtRSF/labelsTr/<case>.nii.gz

Output:
    <build>/rsfseg/stage_15_display3d/fold_X/left/    (mosaic-PNG's)
    <build>/rsfseg/stage_15_display3d/fold_X/right/

Use:
    python stage_15_display3d.py
    python stage_15_display3d.py --folds 0 2 --sides left
    python stage_15_display3d.py --folds ensemble --gauss 1.5 --smooth 30
    python stage_15_display3d.py --gauss 0 --smooth 0        (no smoothing)
"""

import os
import glob
import argparse

import numpy as np
import nibabel as nib

from utils.display3d import rotating_mosaics_da

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


def run_fold(fold_dir, labels_dir, out_dir, subjects_per_chunk, nviews, columns, n,
             sides, gauss, smooth_iter):
    pred_files = sorted(glob.glob(os.path.join(fold_dir, 'predictions', '*.nii.gz')))
    if n > 0:
        pred_files = pred_files[:n]
    if not pred_files:
        print(f'Geen predicties gevonden in {fold_dir}, overgeslagen')
        return

    for side in sides:
        masks, labels = [], []
        for pred_file in pred_files:
            case = os.path.basename(pred_file)[:-len('.nii.gz')]
            label_file = os.path.join(labels_dir, f'{case}.nii.gz')
            if not os.path.isfile(label_file):
                print(f'Label ontbreekt voor {case}, overgeslagen')
                continue
            masks.append(_LazyMask(label_file, SIDES[side]))
            labels.append(f'{case} {side} ref')
            masks.append(_LazyMask(pred_file, SIDES[side]))
            labels.append(f'{case} {side} pred')

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
        nviews=25, columns=4, sides=('left', 'right'), gauss=1.0, smooth_iter=20):
    if columns % 2 != 0:
        raise ValueError('columns moet even zijn zodat ref en pred naast elkaar staan.')

    build_rsf = os.path.join(build, PIPELINE)
    stage13 = os.path.join(build_rsf, 'stage_13_test')
    labels_dir = os.path.join(build_rsf, 'stage_8_build_training_data', DATASET, 'labelsTr')
    output = os.path.join(build_rsf, STAGE)

    for fold in folds:
        name = 'ensemble' if fold == 'ensemble' else f'fold_{fold}'
        run_fold(os.path.join(stage13, name), labels_dir, os.path.join(output, name),
                 subjects_per_chunk, nviews, columns, n, sides, gauss, smooth_iter)


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
    args = parser.parse_args()

    run(args.build, folds=tuple(args.folds), n=args.n,
        subjects_per_chunk=args.per_chunk, nviews=args.nviews, columns=args.columns,
        sides=tuple(args.sides), gauss=args.gauss, smooth_iter=args.smooth)