"""Stage 13: test the packaged RSF model from stage 12, fold by fold.

The model is loaded directly with nnUNetPredictor from the folder that stage 12
built (not via miblab_dl), so the weights that are tested are the ones that
were trained in stage 10.

For every fold, the weights of that fold are applied to the validation cases of
that same fold (taken from splits_final.json of stage 9). Those cases were not
used to train that fold, so the Dice scores are an honest check that the
packaged weights work, and they can be compared fold by fold.

With --ensemble the 5 folds are additionally combined on the first --n cases of
fold 0. Those cases are part of the training data of 4 of the 5 folds, so those
scores are optimistic and only show that the ensemble runs end to end.

Resume: the script can be restarted after it was interrupted.
    - A fold whose results.csv is complete is skipped entirely.
    - Cases that already have a prediction file are not predicted again (the
      Dice is recomputed from the file on disk).
    - Overlay pngs that already exist are not made again.
To force a full redo, delete the fold folder (or its results.csv / predictions).

Input: the nifti files that stage 8 wrote (5 channels: out_phase, in_phase,
water, fat, fat fraction) and the labels (left=1, right=2).

Output (in rsfseg/stage_13_test):
    results.csv              Dice per fold, case and class (all folds merged)
    fold_X/results.csv       Dice of one fold (written when the fold is done)
    fold_X/predictions/      predicted label maps
    fold_X/*.png             mosaic overlays of label and prediction
"""

import os
import csv
import json
import logging
import argparse
from functools import partial

import numpy as np
import nibabel as nib
from miblab_plot import mosaic_overlay

STAGE = 'stage_13_test'
PIPELINE = 'rsfseg'
DATASET = 'Dataset001_iBEAtRSF'
PACKAGED = 'Dataset001_RSFPredictor'
TRAINER = 'nnUNetTrainer__nnUNetPlans__3d_fullres'
N_CHANNELS = 5
FAT = 3  # channel index of the fat image, used as overlay background
CLASS_MAP = {1: "renal_sinus_fat_left", 2: "renal_sinus_fat_right"}
CSV_HEADER = ['fold', 'case', 'class', 'dice']


def dice(a, b):
    """Dice overlap between two masks (NaN if both are empty)."""
    a, b = a.astype(bool), b.astype(bool)
    denom = a.sum() + b.sum()
    return 2 * (a & b).sum() / denom if denom else np.nan


def save_overlay(background, labels, png):
    """Save a mosaic overlay of an RSF label map on a background image."""
    background = background.astype(np.float32)
    background = background / max(np.percentile(background, 99), 1e-6)
    rois = {roi: (labels == idx).astype(np.int16) for idx, roi in CLASS_MAP.items()}
    mosaic_overlay(background, rois, png, vmin=0, vmax=1.0, margin=[16, 16, 2], opacity=0.75)


def load_predictor(model_folder, folds):
    """Load the packaged model from stage 12 for the given folds."""
    # nnunetv2 is imported here, after the environment variables are set in run()
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=(device.type == 'cuda'),
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=folds,
        checkpoint_name='checkpoint_final.pth',  # stage 12 packages checkpoint_final
    )
    return predictor


def nnunet_validation_dice(train_fold_folder):
    """Mean Dice per class that nnU-Net itself computed at the end of training.

    Best effort: returns None if the summary is missing or has another layout.
    """
    path = os.path.join(train_fold_folder, 'validation', 'summary.json')
    try:
        with open(path) as fp:
            mean = json.load(fp)['mean']
        return {roi: mean[str(idx)]['Dice'] for idx, roi in CLASS_MAP.items()}
    except Exception as e:
        logging.warning(f'Could not read nnU-Net validation summary {path}: {e}')
        return None


def read_done_fold(fold_csv, cases):
    """Return the rows of a finished fold, or None if the fold is not complete."""
    if not os.path.isfile(fold_csv):
        return None
    try:
        with open(fold_csv, newline='') as fp:
            rows = list(csv.reader(fp))[1:]
    except Exception as e:
        logging.warning(f'Could not read {fold_csv}: {e}')
        return None
    if len(rows) == len(cases) * len(CLASS_MAP) and {r[1] for r in rows} == set(cases):
        return rows
    return None


def test_cases(get_predictor, cases, images, labels, out_dir, rows, tag):
    """Predict the cases, compare with the labels and save overlays.

    `get_predictor` is a function that loads the model; it is only called if
    there is something left to predict. Appends one row per case and class to
    `rows` and returns the Dice scores per class.
    """
    os.makedirs(out_dir, exist_ok=True)
    pred_dir = os.path.join(out_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    fold_csv = os.path.join(out_dir, 'results.csv')
    scores = {roi: [] for roi in CLASS_MAP.values()}

    # Whole fold finished in an earlier run: reuse its results
    done = read_done_fold(fold_csv, cases)
    if done is not None:
        logging.info(f'{tag}: already complete ({len(cases)} cases), skipping')
        for r in done:
            rows.append(r)
            scores[r[2]].append(float(r[3]))
        return scores

    files = [[os.path.join(images, f'{c}_000{i}.nii.gz') for i in range(N_CHANNELS)] for c in cases]
    for group in files:
        for f in group:
            if not os.path.isfile(f):
                raise FileNotFoundError(f'Missing input file: {f}')

    def pred_path(case):
        return os.path.join(pred_dir, f'{case}.nii.gz')

    # Only predict the cases that have no prediction yet
    todo = [i for i, c in enumerate(cases) if not os.path.isfile(pred_path(c))]
    logging.info(f'{tag}: {len(cases) - len(todo)} predictions already on disk, predicting {len(todo)}')
    if todo:
        predictor = get_predictor()
        predictor.predict_from_files(
            [files[i] for i in todo],
            [os.path.join(pred_dir, cases[i]) for i in todo],
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )

    fold_rows = []
    failed = []
    for case, group in zip(cases, files):
        try:
            truth = np.rint(nib.load(os.path.join(labels, f'{case}.nii.gz')).get_fdata()).astype(np.int16)
            pred = np.rint(nib.load(pred_path(case)).get_fdata()).astype(np.int16)
        except Exception as e:
            # e.g. a prediction file that was cut off when the job was killed
            logging.error(f'{tag} {case}: could not read label/prediction ({e}); prediction removed, rerun to redo it')
            if os.path.isfile(pred_path(case)):
                os.remove(pred_path(case))
            failed.append(case)
            continue
        if truth.shape != pred.shape:
            logging.error(f'{tag} {case}: shape mismatch, predicted {pred.shape} vs label {truth.shape}')
            failed.append(case)
            continue
        for idx, roi in CLASS_MAP.items():
            d = dice(pred == idx, truth == idx)
            scores[roi].append(d)
            fold_rows.append([tag, case, roi, f'{d:.4f}'])
            logging.info(f'{tag} {case}: Dice {roi}: {d:.3f}')

        png_label = os.path.join(out_dir, f'{case}__label.png')
        png_pred = os.path.join(out_dir, f'{case}__predicted.png')
        if not (os.path.isfile(png_label) and os.path.isfile(png_pred)):
            fat = nib.load(group[FAT]).get_fdata()
            if not os.path.isfile(png_label):
                save_overlay(fat, truth, png_label)
            if not os.path.isfile(png_pred):
                save_overlay(fat, pred, png_pred)

    rows.extend(fold_rows)

    # Mark the fold as finished only if every case was scored
    if failed:
        logging.error(f'{tag}: {len(failed)} cases failed ({", ".join(failed)}); fold not marked as complete')
    else:
        with open(fold_csv, 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(CSV_HEADER)
            writer.writerows(fold_rows)
        logging.info(f'{tag}: fold complete, results written to {fold_csv}')

    return scores


def log_means(tag, scores):
    for roi, values in scores.items():
        if values:
            logging.info(f'{tag}: mean Dice {roi}: {np.nanmean(values):.3f} over {len(values)} cases')


def run(build, n=-1, folds=(0, 1, 2, 3, 4), ensemble=False):

    build_rsf = os.path.join(build, PIPELINE)
    raw = os.path.join(build_rsf, 'stage_8_build_training_data', DATASET)
    splits_file = os.path.join(build_rsf, 'stage_9_preprocess', DATASET, 'splits_final.json')
    train_folder = os.path.join(build_rsf, 'stage_10_train', DATASET, TRAINER)
    model_folder = os.path.join(
        build_rsf, 'stage_12_build_contribution', 'RSFPredictor', PACKAGED, TRAINER,
    )
    output = os.path.join(build_rsf, STAGE)
    os.makedirs(output, exist_ok=True)

    # nnU-Net wants these to exist; they are not used when predicting from a folder
    for var in ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']:
        os.environ[var] = output

    for path in [model_folder, splits_file]:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Not found: {path}')
    for f in folds:
        if not os.path.isfile(os.path.join(model_folder, f'fold_{f}', 'checkpoint_final.pth')):
            raise FileNotFoundError(f'Packaged model has no checkpoint for fold {f}')
    with open(splits_file) as fp:
        splits = json.load(fp)

    images = os.path.join(raw, 'imagesTr')
    labels = os.path.join(raw, 'labelsTr')
    rows = []

    # One fold at a time: weights of fold k on the validation cases of fold k
    for fold in folds:
        val_cases = splits[fold]['val']
        cases = val_cases if n <= 0 else val_cases[:n]
        tag = f'fold_{fold}'
        logging.info(f'{tag}: testing {len(cases)} of {len(val_cases)} validation cases')

        scores = test_cases(partial(load_predictor, model_folder, (fold,)),
                            cases, images, labels, os.path.join(output, tag), rows, tag)
        log_means(tag, scores)

        # Reference: what nnU-Net itself found on all validation cases of this fold
        reference = nnunet_validation_dice(os.path.join(train_folder, tag))
        if reference:
            for roi, value in reference.items():
                logging.info(f'{tag}: nnU-Net validation mean Dice {roi} (all val cases): {value:.3f}')

    # Optional: all folds combined (optimistic, see module docstring)
    if ensemble:
        first = folds[0]
        val_cases = splits[first]['val']
        cases = val_cases if n <= 0 else val_cases[:n]
        scores = test_cases(partial(load_predictor, model_folder, tuple(range(5))),
                            cases, images, labels, os.path.join(output, 'ensemble'), rows, 'ensemble')
        log_means('ensemble (optimistic)', scores)

    with open(os.path.join(output, 'results.csv'), 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
    logging.info(f'Results written to {os.path.join(output, "results.csv")}')


if __name__ == '__main__':

    BUILD = r"X:\abdominal_imaging\Shared\Benthe\iBEAt_Build"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    parser.add_argument("--n", type=int, default=3,
                        help="Validation cases to test per fold (0 = all)")
    parser.add_argument("--folds", type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help="Folds to test, e.g. --folds 0 2")
    parser.add_argument("--ensemble", action="store_true",
                        help="Also run all 5 folds combined (scores are optimistic)")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.build, PIPELINE), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.build, PIPELINE, f'{STAGE}.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    run(args.build, n=args.n, folds=tuple(args.folds), ensemble=args.ensemble)