import os
import shutil
import subprocess
import argparse
import logging


def run(build, dataset_id="1"):

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



if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"

    # Comment for the cluster
    os.environ['nnUNet_n_proc_DA'] = '4' # Set in .sh file
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    parser.add_argument("--dataset", type=str, default="1", help="nnU-Net dataset ID")
    args = parser.parse_args()

    rsf_build = os.path.join(args.build, 'rsfseg')
    os.makedirs(rsf_build, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(rsf_build, 'stage_9_preprocess.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    run(args.build, args.dataset)