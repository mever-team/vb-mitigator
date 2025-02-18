#!/bin/bash
## Resource Request
#SBATCH -c8
#SBATCH --mem=12G
#SBATCH --gres shard:12
#SBATCH --job-name="mavias-all"
#SBATCH --output=slurm/slurm_%x_%A_%a.out
#SBATCH --error=slurm/slurm_%x_%A_%a.err
#SBATCH --array=0-79             
#SBATCH --time=0-50:00:00


# Environment setup
export PYTHONPATH="/mnt/cephfs/home/gsarridis/projects/vb-mitigator/"
cd /mnt/cephfs/home/gsarridis/projects/vb-mitigator

conda activate dl310

# Define datasets and methods
DATASETS=("celeba" "imagenet9")
METHODS=("mavias" "debian" "flacb" "jtt" "lff" "sd" "softcon" "erm")
SEEDS=(0 1 2 3 4)

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
if [ "$DATASET" == "celeba" ]; then
    CONFIG_PATH="configs/$DATASET/$METHOD/blonde.yaml"
else
    CONFIG_PATH="configs/$DATASET/$METHOD/dev.yaml"
fi

# Log system info
echo "====================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPUs Allocated: $CUDA_VISIBLE_DEVICES"
echo "Running experiment: Dataset=$DATASET, Method=$METHOD, Seed=$SEED"
echo "====================================="

srun python tools/train.py --cfg "$CONFIG_PATH" --seed "$SEED"