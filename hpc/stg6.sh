#!/bin/bash
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mail-user=bariens1@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=stg6_measure
#SBATCH --output=logs/stg6_measure.out
#SBATCH --error=logs/stg6_measure.err

# NOTE: the logs/ folder must exist before you submit (run `mkdir -p logs`
# once in the folder you call sbatch from); Slurm does not create it.

# Unsets the CPU binding policy.
# Some clusters automatically bind threads to cores; unsetting it can
# prevent performance issues if your code manages threading itself
# (e.g. OpenMP, NumPy, or PyTorch).
unset SLURM_CPU_BIND
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
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

# Copy the stage 4 output to scratch. Trailing-slash-free source with the
# PARENT folder as destination, so it lands in $BUILD/rsfseg/stage_4_edit.
# index.json is excluded: dbdicom rebuilds it on this machine.

rsync -av --no-group --no-perms --ignore-existing --exclude='index.json' \
    "$ARCHIVE/rsfseg/stage_4_edit" "$BUILD/rsfseg/"
    
# srun runs your program on the allocated compute resources managed by Slurm
srun "$ENV/bin/python" "$CODE/stage_6_measure.py" --build="$BUILD"

# Copy results back: the per-image results ...
rsync -av --no-group --no-perms "$BUILD/rsfseg/stage_6_measure" "$ARCHIVE/rsfseg/"
rsync -av --no-group --no-perms --include='*_all_results*' --exclude='*' \
    "$BUILD/rsfseg/" "$ARCHIVE/rsfseg/"