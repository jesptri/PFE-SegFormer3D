#!/bin/bash
set -e

RAW_DATA_DIR="/app/data/brats2017_seg/brats2017_raw_data/train"
PROCESSED_DATA_DIR="/app/data/brats2017_seg/BraTS2017_Training_Data"

echo "Checking raw data..."
if [ ! -d "$RAW_DATA_DIR" ]; then
  echo "Raw data not found at $RAW_DATA_DIR"
  exit 1
fi

# 1. Preprocessing (si nécessaire)
if [ ! -d "$PROCESSED_DATA_DIR" ] || [ -z "$(ls -A $PROCESSED_DATA_DIR)" ]; then
  echo "Running preprocessing..."
  cd /app/data/brats2017_seg/brats2017_raw_data/
  python3 brats2017_seg_preprocess.py
else
  echo "Preprocessed data already exists"
fi

# # 2. Training
# echo "Starting training..."
# accelerate launch \
#   /app/experiments/brats_2017/your_experiment/run_experiment.py

# echo "Done"
