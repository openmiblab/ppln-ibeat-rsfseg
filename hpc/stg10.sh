#!/bin/bash
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=96:00:00
#SBATCH --mail-user=bariens1@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=stg10
#SBATCH --array=1-4
#SBATCH --output=logs/stg10_fold%a.out
#SBATCH --error=logs/stg10_fold%a.err

# Unsets the CPU binding policy.
# Some clusters automatically bind threads to cores; unsetting it can 
# prevent performance issues if your code manages threading itself 
# (e.g. OpenMP, NumPy, or PyTorch).
unset SLURM_CPU_BIND

# Ensures that all your environment variables from the submission 
# environment are passed into the job's environment
export SLURM_EXPORT_ENV=ALL

# Loads the Anaconda module provided by the cluster.
# (On HPC systems, software is usually installed as "modules" to avoid version conflicts.)
module load Anaconda3/2025.06-1
module load Python/3.10.8-GCCcore-12.2.0 # essential to load latest GCC

# Tell VTK/PyVista to use the OSMesa (Off-Screen) library
export VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow

# Define path variables here
ENV="/mnt/parscratch/users/$(whoami)/envs/rsfseg"
CODE="/mnt/parscratch/users/$(whoami)/scripts/ppln-ibeat-rsfseg/src/ibeat_rsfseg"
BUILD="/mnt/parscratch/users/$(whoami)/data/iBEAt_Build"
ARCHIVE="logina:/shared/abdominal_imaging/Shared/Benthe/iBEAt_Build"

# Each array task trains one fold (1, 2, 3, 4). Fold 0 is already done.
FOLD=$SLURM_ARRAY_TASK_ID

# Path of the fold inside the nnU-Net results folder
MODEL="Dataset001_iBEAtRSF/nnUNetTrainer__nnUNetPlans__3d_fullres"

# srun runs your program on the allocated compute resources managed by Slurm
srun "$ENV/bin/python" "$CODE/stage_10_train.py" --build="$BUILD" --fold=$FOLD

# Copy back only this job's fold, so parallel jobs don't write to the same folder
rsync -av --no-group --no-perms \
    "$BUILD/rsfseg/stage_10_train/$MODEL/fold_$FOLD" \
    "$ARCHIVE/rsfseg/stage_10_train/$MODEL/"