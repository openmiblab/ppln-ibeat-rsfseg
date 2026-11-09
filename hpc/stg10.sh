#!/bin/bash
#SBATCH --mem=70G
#SBATCH --cpus-per-task=16
#SBATCH --qos=gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=96:00:00
#SBATCH --mail-user=bariens1@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=stg10
#SBATCH --array=0-4
#SBATCH --output=logs/stg10_fold%a.out
#SBATCH --error=logs/stg10_fold%a.err
#SBATCH --exclude=gpu17 

# Unsets the CPU binding policy.
# Some clusters automatically bind threads to cores; unsetting it can 
# prevent performance issues if your code manages threading itself 
# (e.g. OpenMP, NumPy, or PyTorch).
unset SLURM_CPU_BIND

export SLURM_EXPORT_ENV=ALL

module load Anaconda3/2025.06-1
module load Python/3.10.8-GCCcore-12.2.0 # essential to load latest GCC

# Tell VTK/PyVista to use the OSMesa (Off-Screen) library
export VTK_DEFAULT_OPENGL_WINDOW=vtkOSOpenGLRenderWindow

# Define path variables here
ENV="/mnt/parscratch/users/$(whoami)/envs/rsfseg"
CODE="/mnt/parscratch/users/$(whoami)/scripts/ppln-ibeat-rsfseg/src/ibeat_rsfseg"
BUILD="/mnt/parscratch/users/$(whoami)/data/iBEAt_Build"
ARCHIVE="loginb:/shared/abdominal_imaging/Shared/Benthe/iBEAt_Build"

FOLD=$SLURM_ARRAY_TASK_ID
MODEL="Dataset001_iBEAtRSF/nnUNetTrainer__nnUNetPlans__3d_fullres"
nvidia-smi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
"$ENV/bin/python" -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

srun "$ENV/bin/python" "$CODE/stage_10_train.py" --build="$BUILD" --fold=$FOLD

rsync -av --no-group --no-perms \
    "$BUILD/rsfseg/stage_10_train/$MODEL/fold_$FOLD" \
    "$ARCHIVE/rsfseg/stage_10_train/$MODEL/"