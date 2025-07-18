#!/bin/bash
#SBATCH -c1
#SBATCH --mem=6G
#SBATCH --gres shard:4
#SBATCH --job-name="biasinbios"
#SBATCH --output=slurm/slurm_%x_%A_%a.out
#SBATCH --error=slurm/slurm_%x_%A_%a.err
#SBATCH --array=0-43   # 1 methods × 1 seeds × 9 runs 
#SBATCH --time=06:00:00  # 3-hour slots



# Environment setup
export PYTHONPATH="/mnt/cephfs/home/gsarridis/projects/vb-mitigator/"
cd /mnt/cephfs/home/gsarridis/projects/vb-mitigator

conda activate dl310


# Define datasets and methods
DATASETS=("bias_in_bios")
METHODS=("erm" "debian" "jtt" "lff" "sd" "di" "badd" "flac" "bb" "end" "groupdro")
SEEDS=(1 2 3 4)
# Compute indices from SLURM_ARRAY_TASK_ID
TOTAL_EXPERIMENTS=$(( ${#DATASETS[@]} * ${#METHODS[@]} * ${#SEEDS[@]} ))
EXPERIMENT_ID=$SLURM_ARRAY_TASK_ID

# Compute dataset, method, and seed indices
DATASET_IDX=$(( EXPERIMENT_ID / (${#METHODS[@]} * ${#SEEDS[@]}) ))
METHOD_IDX=$(( (EXPERIMENT_ID / ${#SEEDS[@]}) % ${#METHODS[@]} ))
SEED_IDX=$(( EXPERIMENT_ID % ${#SEEDS[@]} ))


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

srun python tools/train.py --cfg "$CONFIG_PATH" --seed "$SEED"