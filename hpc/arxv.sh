#!/bin/bash   
#SBATCH --mem=4G         
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mail-user=bariens1@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=arxv
#SBATCH --output=logs/arxv.out
#SBATCH --error=logs/arxv.err

# Unsets the CPU binding policy.
# Some clusters automatically bind threads to cores; unsetting it can 
# prevent performance issues if your code manages threading itself 
# (e.g. OpenMP, NumPy, or PyTorch).
unset SLURM_CPU_BIND

# Ensures that all your environment variables from the submission 
# environment are passed into the job’s environment
export SLURM_EXPORT_ENV=ALL

# Loads the Anaconda module provided by the cluster.
# (On HPC systems, software is usually installed as “modules” to avoid version conflicts.)
module load Anaconda3/2024.02-1
module load Python/3.10.8-GCCcore-12.2.0 # essential to load latest GCC

# Define path variables here
HPC="/mnt/parscratch/users/$(whoami)/data/iBEAt_Build/dixon"
ARCHIVE="login1:/shared/abdominal_imaging/Archive/iBEAt_Build/dixon/stage_5_clean_dixon_data"

# srun runs your program on the allocated compute resources managed by Slurm
rsync -av --no-group --no-perms "$ARCHIVE" "$HPC" 

# Define path variables here
HPC="/mnt/parscratch/users/$(whoami)/data/iBEAt_Build/kidneyvol"
ARCHIVE="login1:/shared/abdominal_imaging/Archive/iBEAt_Build/kidneyvol/stage_3_edit"

# srun runs your program on the allocated compute resources managed by Slurm
rsync -av --no-group --no-perms "$ARCHIVE" "$HPC"