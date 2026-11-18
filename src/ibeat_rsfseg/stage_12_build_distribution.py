import os
import shutil
import argparse
import logging

logger = logging.getLogger(__name__)

STAGE = 'stage_12_build_contribution'
FOLDS = [0, 1, 2, 3, 4]
TRAINER = 'nnUNetTrainer__nnUNetPlans__3d_fullres'


def copy_file(src, dst_folder):
    """Copy a single file into dst_folder, failing loudly if it is missing."""
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Required file not found: {src}")
    os.makedirs(dst_folder, exist_ok=True)
    shutil.copy(src, dst_folder)
    logger.info(f"Copied {src} -> {dst_folder}")


def run(build):
    """Build a lightweight folder with bare model weights for contribution."""

    # Input: trained model from the training stage
    source = os.path.join(
        build, 'rsfseg', 'stage_10_train', 'Dataset001_iBEAtRSF', TRAINER,
    )

    # Output: contribution folder for this stage
    destination = os.path.join(
        build, 'rsfseg', STAGE, 'RSFPredictor', 'Dataset001_RSFPredictor', TRAINER,
    )
    os.makedirs(destination, exist_ok=True)

    # Header files
    for name in ['dataset.json', 'dataset_fingerprint.json', 'plans.json']:
        copy_file(os.path.join(source, name), destination)

    # Weights
    for fold in FOLDS:
        copy_file(
            os.path.join(source, f'fold_{fold}', 'checkpoint_final.pth'),
            os.path.join(destination, f'fold_{fold}'),
        )

    # Files needed for postprocessing
    crossval = "crossval_results_folds_0_1_2_3_4"
    for name in ['postprocessing.pkl', 'plans.json']:
        copy_file(
            os.path.join(source, crossval, name),
            os.path.join(destination, crossval),
        )

    logger.info(f"{STAGE} finished: {destination}")


if __name__ == '__main__':

    BUILD = r"X:\abdominal_imaging\Shared\Benthe\iBEAt_Build"

    # Comment for the cluster
    os.environ['nnUNet_n_proc_DA'] = '4'  # Set in .sh file
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.build, 'rsfseg'), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.build, 'rsfseg', f'{STAGE}.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    run(args.build)