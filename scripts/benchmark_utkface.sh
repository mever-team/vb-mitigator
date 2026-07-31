#!/usr/bin/env bash
# Reproduce the UTKFace (race bias) benchmark: run every available method with
# its config and 3 seeds. Template — copy for other datasets/benchmarks.
set -euo pipefail

DATASET="utkface"
SEEDS=(1 2 3)

for cfg in configs/${DATASET}/*/*.yaml; do
  for seed in "${SEEDS[@]}"; do
    echo "=== ${cfg} (seed ${seed}) ==="
    vbm-train --cfg "${cfg}" EXPERIMENT.SEED "${seed}"
  done
done
