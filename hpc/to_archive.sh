#!/bin/bash
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mail-user=s.sourbron@sheffield.ac.uk
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=to_archive
#SBATCH --output=logs/to_archive.out
#SBATCH --error=logs/to_archive.err

# This ensure that the second command is only run if the first succeeds
set -euo pipefail

SHARED="logina:/shared/abdominal_imaging/Shared/Benthe/iBEAt_Build"
ARCHIVE="logina:/shared/abdominal_imaging/Archive/iBEAt_Build"
STAGING="/mnt/parscratch/users/$(whoami)/data/iBEAt_Build"

echo "Step 1: shared -> staging"
rsync -av --partial --no-group --no-perms --delete "$SHARED/rsfseg" "$STAGING"

echo "Step 2: staging -> archive"
rsync -av --partial --no-group --no-perms --delete "$STAGING/rsfseg" "$ARCHIVE"

echo "Done"

# Check this before deleting the staging folder
# rsync -avnc --no-group --no-perms "$STAGING/rsfseg" "$ARCHIVE"