import os
import json
import shutil
import subprocess
import argparse
import logging
import random

from utils import data


DATASET_NAME = 'Dataset001_iBEAtRSF'
N_FOLDS = 5
SEED = 42


def make_patient_splits(cases, case_to_patient, n_folds=N_FOLDS, seed=SEED):
    """Group k-fold: all cases of one patient/volunteer end up in the same fold.

    Patients are shuffled (fixed seed, so reproducible) and assigned one by
    one to the fold that currently has the fewest cases, which keeps the
    folds balanced in size even if some patients have several visits.
    """
    by_patient = {}
    for case in cases:
        by_patient.setdefault(case_to_patient[case], []).append(case)

    patients = sorted(by_patient)
    random.Random(seed).shuffle(patients)
    # Biggest groups first so the balancing works well
    patients.sort(key=lambda p: -len(by_patient[p]))

    folds = [[] for _ in range(n_folds)]
    for patient in patients:
        smallest = min(range(n_folds), key=lambda k: len(folds[k]))
        folds[smallest].extend(by_patient[patient])

    splits = []
    for k in range(n_folds):
        val = sorted(folds[k])
        train = sorted(c for j in range(n_folds) if j != k for c in folds[j])
        splits.append({'train': train, 'val': val})
    return splits


def write_splits(raw_dataset, preproc_dataset, results_dataset, force=False):
    """Write a patient-grouped splits_final.json for nnU-Net.

    Must happen BEFORE training. If training results already exist the
    split is not touched unless force=True, because changing the split
    after a fold has been trained makes the folds inconsistent.
    """
    splits_file = os.path.join(preproc_dataset, 'splits_final.json')

    started = os.path.isdir(results_dataset) and any(
        d.startswith('fold_')
        for _, dirs, _ in os.walk(results_dataset) for d in dirs
    )
    if started and not force:
        msg = (
            f'Training results already exist in {results_dataset}. '
            'Not changing splits_final.json (use --force-splits and delete '
            'the trained folds if you want to redo all folds).'
        )
        logging.warning(msg)
        print(msg)
        return

    labels_dir = os.path.join(raw_dataset, 'labelsTr')
    cases = sorted(f[:-7] for f in os.listdir(labels_dir) if f.endswith('.nii.gz'))

    # Same case-id convention as stage 8: f"{patient}_{study}" with spaces -> _
    record = data.dixon_record()
    case_to_patient = {
        f"{row[1]}_{row[2]}".replace(' ', '_'): row[1] for row in record[1:]
    }
    missing = [c for c in cases if c not in case_to_patient]
    if missing:
        raise ValueError(
            f'{len(missing)} cases not found in dixon_data.csv, e.g. {missing[:3]}'
        )

    splits = make_patient_splits(cases, case_to_patient)

    # Sanity check: no patient in both train and val of any fold
    for k, s in enumerate(splits):
        p_train = {case_to_patient[c] for c in s['train']}
        p_val = {case_to_patient[c] for c in s['val']}
        assert not (p_train & p_val), f'fold {k}: patient overlap'

    os.makedirs(preproc_dataset, exist_ok=True)
    with open(splits_file, 'w') as f:
        json.dump(splits, f, indent=2)

    n_patients = len(set(case_to_patient[c] for c in cases))
    msg = (
        f'Wrote patient-grouped splits ({N_FOLDS} folds, {len(cases)} cases, '
        f'{n_patients} patients) to {splits_file}. '
        f'Validation sizes: {[len(s["val"]) for s in splits]}'
    )
    logging.info(msg)
    print(msg)


def run(build, dataset_id="1", force_splits=False):

    build_rsf = os.path.join(build, 'rsfseg')

    raw_data = os.path.join(build_rsf, 'stage_8_build_training_data')
    preproc_data = os.path.join(build_rsf, 'stage_9_preprocess')
    results = os.path.join(build_rsf, 'stage_10_train')

    # Ensure folders exist
    os.makedirs(preproc_data, exist_ok=True)
    os.makedirs(results, exist_ok=True)

    # Define environment variables
    # https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md
    os.environ["nnUNet_raw"] = raw_data
    os.environ["nnUNet_preprocessed"] = preproc_data
    os.environ["nnUNet_results"] = results

    # Find the nnU-Net executable: on PATH if available, else the cluster env
    exe = (
        shutil.which("nnUNetv2_plan_and_preprocess")
        or "/mnt/parscratch/users/cmr26ba/envs/rsfseg/bin/nnUNetv2_plan_and_preprocess"
    )

    # https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md
    cmd = [
        exe,
        "-d", dataset_id,
        "-c", "3d_fullres",
        "--verify_dataset_integrity",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",   # force UTF-8 decoding
        errors="replace"    # avoid crash if weird bytes appear
    )

    # Stream logs in real-time
    for line in process.stdout:
        print(line, end="")

    process.wait()  # wait for completion

    if process.returncode != 0:
        raise RuntimeError(f"nnUNet preprocessing failed (exit code {process.returncode})")

    logging.info('Preprocessing complete..')

    # Patient-grouped cross-validation splits, so that repeat visits of the
    # same patient never end up in both training and validation.
    write_splits(
        os.path.join(raw_data, DATASET_NAME),
        os.path.join(preproc_data, DATASET_NAME),
        os.path.join(results, DATASET_NAME),
        force=force_splits,
    )


if __name__ == '__main__':

    BUILD = r"X:\abdominal_imaging\Shared\Benthe\iBEAt_Build"

    # Comment for the cluster
    os.environ['nnUNet_n_proc_DA'] = '4' # Set in .sh file
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    parser.add_argument("--dataset", type=str, default="1", help="nnU-Net dataset ID")
    parser.add_argument("--force-splits", action="store_true",
                        help="Overwrite splits_final.json even if training has started")
    args = parser.parse_args()

    rsf_build = os.path.join(args.build, 'rsfseg')
    os.makedirs(rsf_build, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(rsf_build, 'stage_9_preprocess.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    run(args.build, args.dataset, force_splits=args.force_splits)