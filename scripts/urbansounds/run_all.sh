#!/bin/bash

# Define your experiment parameters
DATASETS=("urbansounds58")
METHODS=("badd" "flac")
SEEDS=(0 1 2 3 4)

echo "Starting experiments..."

# Loop through each seed
for SEED in "${SEEDS[@]}"; do
    # Loop through each dataset
    for DATASET in "${DATASETS[@]}"; do
        # Loop through each method
        for METHOD in "${METHODS[@]}"; do
        
            echo "----------------------------------------------------"
            echo "Running experiment for Dataset: $DATASET, Method: $METHOD, Seed: $SEED"
            echo "----------------------------------------------------"

            # Construct the configuration file path
            # Make sure this path is correct relative to where you run the script
            CONFIG_PATH="configs/${DATASET}/${METHOD}/dev.yaml"

            # Check if the config file exists before running
            if [ -f "$CONFIG_PATH" ]; then
                # Run the Python training script
                # Ensure 'python' points to the correct interpreter (e.g., python3 or a virtual env)
                python tools/train.py --cfg "$CONFIG_PATH" --seed "$SEED"

                # Check the exit status of the last command (python script)
                if [ $? -eq 0 ]; then
                    echo "Experiment for $DATASET / $METHOD / Seed $SEED completed successfully."
                else
                    echo "Error: Experiment for $DATASET / $METHOD / Seed $SEED failed!"
                    # You might want to add error handling here, e.g., exit or log to a specific file
                fi
            else
                echo "Warning: Config file not found: $CONFIG_PATH. Skipping this combination."
            fi
            echo "" # Add an empty line for readability between runs
        done
    done
done

echo "All experiments finished."