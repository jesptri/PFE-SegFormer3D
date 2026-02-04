#!/bin/bash
set -e

export PYTHONPATH="$PYTHONPATH:$(pwd)"

SHOW_HELP=false
DO_PREPROCESSING=false
DO_TRAINING=false
DO_INFERENCE=false
DO_EVALUATION=false

while getopts "ptieh" opt; do
  case "$opt" in
    p) DO_PREPROCESSING=true ;;
    t) DO_TRAINING=true ;;
    i) DO_INFERENCE=true ;;
    e) DO_EVALUATION=true ;;
    h) SHOW_HELP=true;;
    \?) echo "*-* Invalid option: -$OPTARG" ;;
  esac
done

if $SHOW_HELP; then
  echo "*-* Entrypoint helper for this Segformer 3D implementation"
  echo "-p [Preprocessing] : allows to run the preprocessing scripts."
  echo "-t [Training] : allows to run the training scripts."
  echo "-i [Inference] : allows to run the inference scripts."
  echo "-e [Evaluation] : allows to run the evaluation scripts."
  echo "-h [Help] : shows this help message."
  echo "*-* End of the helper !"
  exit 0
fi

# paths can be written as it is because they are path inside the docker container
ROOT="."
RAW_DATA_DIR="./data/brats2017_seg/brats2017_raw_data"
TRAIN_DATA_DIR="train"
PROCESSED_DATA_DIR="./data/brats2017_seg/BraTS2017_Training_Data"
EXPERIMENT_DIR=./experiments/brats_2017

# TODO : Maybe add a DO_RESET option to remove the old folder for the processed data

if $DO_PREPROCESSING; then
  echo "*-* Launching the preprocessing..."
  echo "*-* Checking if the raw data folder exists..."
  if [ ! -d "$RAW_DATA_DIR/$TRAIN_DATA_DIR" ]; then
    echo "*-* The raw data folder was not found at $RAW_DATA_DIR/$TRAIN_DATA_DIR"
    exit 1
  else
    echo "*-* The raw data folder was found !"
  fi
  if [ ! -d "$PROCESSED_DATA_DIR" ] || [ -z "$(ls -A $PROCESSED_DATA_DIR)" ]; then
    echo "*-* Running preprocessing script..."
    python3 $RAW_DATA_DIR/brats2017_seg_preprocess.py -r $RAW_DATA_DIR -t $TRAIN_DATA_DIR -o $PROCESSED_DATA_DIR
  else
    echo "*-* Preprocessed data already exists"
  fi
fi

if $DO_TRAINING; then
  cd $EXPERIMENT_DIR/my_experiment
  echo "*-* Running training inside $(pwd)"
  accelerate launch --config_file ./gpu_accelerate.yaml run_experiment.py
  cd - > /dev/null
fi

if $DO_INFERENCE; then
  echo "*-* Not implemented yet."
fi

if $DO_EVALUATION; then
  echo "*-* Launching the evaluation... (only pre-trained model for now)"
  python3 $ROOT/eval_scripts/evaluate.py 
fi
echo "*-* End of the entrypoint"