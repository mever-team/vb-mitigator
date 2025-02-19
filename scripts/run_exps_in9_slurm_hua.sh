#!/bin/bash
## Resource Request
#SBATCH --partition=yoda
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --job-name="mavias-all"
#SBATCH --output=slurm/slurm_%x_%A_%a.out
#SBATCH --error=slurm/slurm_%x_%A_%a.err
#SBATCH --array=0-328    # 8 methods × 5 seeds × 8 runs + MAVIAS (9 feature runs) = 329 jobs
#SBATCH --time=03:00:00  # 3-hour slots


# Environment setup
export PYTHONPATH="/home/isarridis/projects/vb-mitigator/"
cd /home/isarridis/projects/vb-mitigator

conda activate dl310

# Define datasets and methods
DATASETS=("imagenet9")
METHODS=("mavias" "debian" "flacb" "jtt" "lff" "sd" "softcon" "erm")
SEEDS=(0 1 2 3 4)

# Compute total job count
TOTAL_EXPERIMENTS=$(( ${#DATASETS[@]} * ${#METHODS[@]} * ${#SEEDS[@]} * 8 ))  # 8 runs per method
MAVIAS_FEATURE_RUNS=9
MAVIAS_TRAIN_RUNS=8
TOTAL_JOBS=$(( TOTAL_EXPERIMENTS + MAVIAS_FEATURE_RUNS ))

# Compute indices from SLURM_ARRAY_TASK_ID
EXPERIMENT_ID=$SLURM_ARRAY_TASK_ID

if [ "$EXPERIMENT_ID" -lt "$MAVIAS_FEATURE_RUNS" ]; then
    # MAVIAS Feature Extraction (First 9 Jobs)
    STEP=$(( EXPERIMENT_ID + 1 ))
    echo "Running MAVIAS Feature Extraction (Step $STEP/9)"
    srun python tools/compute_features.py --cfg "configs/imagenet9/mavias/dev.yaml"
    exit 0
fi

# Adjust EXPERIMENT_ID for standard training jobs
EXPERIMENT_ID=$(( EXPERIMENT_ID - MAVIAS_FEATURE_RUNS ))

# Compute dataset, method, seed, and run indices
DATASET_IDX=$(( EXPERIMENT_ID / (${#METHODS[@]} * ${#SEEDS[@]} * 8) ))
METHOD_IDX=$(( (EXPERIMENT_ID / (${#SEEDS[@]} * 8)) % ${#METHODS[@]} ))
SEED_IDX=$(( (EXPERIMENT_ID / 8) % ${#SEEDS[@]} ))
RUN_IDX=$(( EXPERIMENT_ID % 8 ))  # 8 training runs

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


echo "====================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPUs Allocated: $CUDA_VISIBLE_DEVICES"
echo "Running experiment: Dataset=$DATASET, Method=$METHOD, Seed=$SEED, Epochs $START_EPOCH-$END_EPOCH"
echo "====================================="

# Run training
srun python tools/train.py --cfg "$CONFIG_PATH" --seed "$SEED" --epoch_steps 5 --placeholder_steps 50000
