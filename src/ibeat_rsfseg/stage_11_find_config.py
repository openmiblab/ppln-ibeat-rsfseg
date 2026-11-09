import os
import sys
import subprocess
import argparse
import logging


def run(build):

    build_rsf = os.path.join(build, 'rsfseg')

    raw_data = os.path.join(build_rsf, 'stage_8_build_training_data')
    preproc_data = os.path.join(build_rsf, 'stage_9_preprocess')
    results = os.path.join(build_rsf, 'stage_10_train')

    os.environ["nnUNet_raw"] = raw_data
    os.environ["nnUNet_preprocessed"] = preproc_data
    os.environ["nnUNet_results"] = results

    find_best = os.path.join(
        os.path.dirname(sys.executable), "nnUNetv2_find_best_configuration"
    )
    cmd = [find_best, "001", "-c", "3d_fullres"]
    logging.info('Started finding best configuration')

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    for line in process.stdout:
        print(line, end="")

    rc = process.wait()
    if rc != 0:
        logging.error(f'find_best_configuration failed with return code {rc}')
        sys.exit(rc)

    logging.info('Finished finding config.')


if __name__ == '__main__':

    BUILD = r"X:\abdominal_imaging\Shared\Benthe\iBEAt_Build"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    logging.basicConfig(
        filename=os.path.join(args.build, 'rsfseg', 'stage_11_find_config.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    run(args.build)