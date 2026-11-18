#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mail-user=bariens1@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=stg13
#SBATCH --output=logs/stg13.out
#SBATCH --error=logs/stg13.err

# Stop at the first error, so rsync never archives half-finished work
set -e

unset SLURM_CPU_BIND
export SLURM_EXPORT_ENV=ALL

module load Anaconda3/2025.06-1
module load Python/3.10.8-GCCcore-12.2.0 # essential to load latest GCC

# Define path variables here
ENV="/mnt/parscratch/users/$(whoami)/envs/rsfseg"
CODE="/mnt/parscratch/users/$(whoami)/scripts/ppln-ibeat-rsfseg/src/ibeat_rsfseg"
BUILD="/mnt/parscratch/users/$(whoami)/data/iBEAt_Build"
ARCHIVE="loginb:/shared/abdominal_imaging/Shared/Benthe/iBEAt_Build"

# Show the GPU in the job log, so a missing GPU is easy to spot
nvidia-smi

# Held-out test per fold: weights of fold k on the validation cases of fold k
srun "$ENV/bin/python" "$CODE/stage_13_test.py" --build="$BUILD" --n=-1

# Predictions and PNGs, plus the log (which sits next to the stage folder)
rsync -av --no-group --no-perms "$BUILD/rsfseg/stage_13_test/" "$ARCHIVE/rsfseg/stage_13_test"
rsync -av --no-group --no-perms "$BUILD/rsfseg/stage_13_test.log" "$ARCHIVE/rsfseg/"