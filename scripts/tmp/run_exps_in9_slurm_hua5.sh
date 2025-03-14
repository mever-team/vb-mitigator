#!/bin/bash
## Resource Request
#SBATCH --partition=yoda
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --job-name="in9-lff"
#SBATCH --output=slurm/slurm_%x_%A_%a.out
#SBATCH --error=slurm/slurm_%x_%A_%a.err
#SBATCH --array=0-17%1    # 1 methods × 1 seeds × 9 runs 
#SBATCH --time=03:00:00  # 3-hour slots


# Environment setup
export PYTHONPATH="/home/isarridis/projects/vb-mitigator/"
cd /home/isarridis/projects/vb-mitigator

source /home/isarridis/anaconda3/etc/profile.d/conda.sh
conda activate dl310

# Define datasets and methods
DATASETS=("imagenet9")
METHODS=("lff")
SEEDS=(0 1 2)
STEPS=6

# Compute indices from SLURM_ARRAY_TASK_ID
TOTAL_EXPERIMENTS=$(( ${#DATASETS[@]} * ${#METHODS[@]} * ${#SEEDS[@]} * $STEPS ))

EXPERIMENT_ID=$SLURM_ARRAY_TASK_ID

# Compute dataset, method, and seed indices
DATASET_IDX=$(( EXPERIMENT_ID / (${#METHODS[@]} * ${#SEEDS[@]} * $STEPS) ))

METHOD_IDX=$(( (EXPERIMENT_ID / (${#SEEDS[@]}* $STEPS)) % ${#METHODS[@]} ))

SEED_IDX=$(( (EXPERIMENT_ID / $STEPS) % ${#SEEDS[@]} ))


# Get corresponding values
DATASET=${DATASETS[$DATASET_IDX]}
METHOD=${METHODS[$METHOD_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# Set config file path
CONFIG_PATH="configs/$DATASET/$METHOD/dev.yaml"
# Log system info
echo "====================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPUs Allocated: $CUDA_VISIBLE_DEVICES"
echo "Running experiment: Dataset=$DATASET, Method=$METHOD, Seed=$SEED"
echo "====================================="


# Run training
srun python tools/train.py --cfg "$CONFIG_PATH" --seed "$SEED" --epoch_steps 5 
