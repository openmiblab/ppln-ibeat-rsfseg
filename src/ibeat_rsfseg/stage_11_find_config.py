import os
import subprocess
import argparse
import logging



def run(build):

    build_rsf = os.path.join(build, 'rsfseg')

    raw_data = os.path.join(build_rsf, 'stage_8_build_training_data')
    preproc_data = os.path.join(build_rsf, 'stage_9_preprocess')
    results = os.path.join(build_rsf, 'stage_10_train') 
   
    # Define environment variables
    # https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md
    os.environ["nnUNet_raw"] = raw_data
    os.environ["nnUNet_preprocessed"] = preproc_data
    os.environ["nnUNet_results"] = results

    # Find best configuration    
    cmd = [
        "nnUNetv2_find_best_configuration",
        "001",
        "-c",
        "3d_fullres"
    ]

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

    logging.info(f'Finished finding config..')



if __name__ == '__main__':

    BUILD = r"X:\abdominal_imaging\Shared\Benthe\iBEAt_Build"

    # Comment for the cluster
    os.environ['nnUNet_n_proc_DA'] = '4' # Set in .sh file
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, default=BUILD, help="Build folder")
    args = parser.parse_args()

    logging.basicConfig(
        filename=os.path.join(args.build, 'rsfseg', 'stage_11_find_config.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    run(args.build)


