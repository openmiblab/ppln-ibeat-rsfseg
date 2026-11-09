#!/bin/bash
#SBATCH --mem=60G
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mail-user=bariens1@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=test
#SBATCH --output=logs/test.out
#SBATCH --error=logs/test.err


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
ARCHIVE="loginb:/shared/abdominal_imaging/Shared/Benthe/iBEAt_Build"

   
srun "$ENV/bin/python" "$CODE/test.py" --build="$BUILD"

# Copy results back: the per-image results ...
