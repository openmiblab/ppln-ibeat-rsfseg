import os
import shutil
import subprocess
import argparse
import logging
import sys


def run(build, fold=0):

    build_rsf = os.path.join(build, 'rsfseg')

    raw_data = os.path.join(build_rsf, 'stage_8_build_training_data')
    preproc_data = os.path.join(build_rsf, 'stage_9_preprocess')
    results = os.path.join(build_rsf, 'stage_10_train')

    # Define environment variables
    # https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md
    os.environ["nnUNet_raw"] = raw_data
    os.environ["nnUNet_preprocessed"] = preproc_data
    os.environ["nnUNet_results"] = results

    # Half the number of CPU cores
    # os.environ['nnUNet_n_proc_DA'] = '4' # Set in .sh file

    # https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md

    nnunet_train = os.path.join(os.path.dirname(sys.executable), "nnUNetv2_train")
    cmd = [nnunet_train, "001", "3d_fullres", str(fold), "--c"]
    logging.info(f'Started training fold {fold}')

    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        encoding="utf-8",   # <-- force UTF-8 decoding
        errors="replace"    # <-- avoids crash if weird bytes appear
    )

    # Stream logs in real-time
    for line in process.stdout:
        print(line, end="")

    process.wait()  # wait for completion

    logging.info(f'Finished training fold {fold}')



if __name__ == '__main__':

    BUILD = r"C:\Users\md1spsx\Documents\Data\iBEAt_Build"

    # Comment for the cluster
    os.environ['nnUNet_n_proc_DA'] = '4' # Set in .sh file
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    parser.add_argument("--fold", type=int, default=0, help="Which fold")
    args = parser.parse_args()

    rsf_build = os.path.join(args.build, 'rsfseg')
    os.makedirs(rsf_build, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(rsf_build, 'stage_10_train.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    run(args.build, fold=args.fold)
