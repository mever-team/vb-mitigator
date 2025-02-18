#!/bin/bash
## Resource Request
#SBATCH -c8                      # Number of CPU cores
#SBATCH --mem=12G                 # Memory request
#SBATCH --gres shard:12
#SBATCH --job-name="in9-erm"
#SBATCH --output=slurm/slurm_%x_%A.out  # Standard output
#SBATCH --error=slurm/slurm_%x_%A.err   # Standard error
#SBATCH --time=0-48:00:00         # Time limit (DD-HH:MM:SS)

# Environment setup
export PYTHONPATH="/mnt/cephfs/home/gsarridis/projects/vb-mitigator/"
cd /mnt/cephfs/home/gsarridis/projects/vb-mitigator

conda activate dl310  # Activate the appropriate conda environment

# Run the experiment
srun python tools/train.py --cfg configs/imagenet9/erm/tags.yaml
srun python tools/train.py --cfg configs/imagenet9/erm/dev.yaml