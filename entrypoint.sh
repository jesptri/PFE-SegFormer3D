#!/bin/bash
set -e

# Silence warnings
export PYTHONWARNINGS="ignore::UserWarning"
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# paths can be written as it is because they are path inside the docker container
ROOT="/app"
RAW_DATA_DIR="/app/data/brats2017_seg/brats2017_raw_data"
OUTPUT_DIR="/app/data/output"
TRAIN_DATA_DIR="train"
PROCESSED_DATA_DIR="/app/data/brats2017_seg/BraTS2017_Training_Data"
EXPERIMENT_DIR=/app/experiments/brats_2017

# get args
SHOW_HELP=false
DO_PREPROCESSING=false
DO_TRAINING=false
DO_INFERENCE=false
declare -i TYPE_INFERENCE
DO_EVALUATION=false
declare -i TYPE_EVALUATION
MODEL_WEIGHTS=-2
while getopts "pti:e:w:h" opt; do
  case "$opt" in
    p) DO_PREPROCESSING=true ;;
    t) DO_TRAINING=true ;;
    i) DO_INFERENCE=true ; TYPE_INFERENCE=$OPTARG ;;
    e) DO_EVALUATION=true ; TYPE_EVALUATION=$OPTARG ;;
    w) MODEL_WEIGHTS=$OPTARG ;;
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
  echo "-w [Weights] : path to model weights file (.pth). Optional."
  echo "-h [Help] : shows this help message."
  echo "*-* End of the helper !"
  exit 0
fi

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
  # create experiment name
  EXPERIMENT_NAME=experiment_$(TZ="Etc/GMT-1" date +"%Y_%m_%d_%H%M") 
  # replace data in config.yaml
  cp -r $EXPERIMENT_DIR/template_experiment $EXPERIMENT_DIR/$EXPERIMENT_NAME
  sed -i "s|__EXPERIMENT_NAME__|${OUTPUT_DIR}/${EXPERIMENT_NAME}|g" "${EXPERIMENT_DIR}/${EXPERIMENT_NAME}/config.yaml"
  # store last experiment name in OUTPUT DIR
  > $OUTPUT_DIR/"last_experiment.txt" 
  echo $EXPERIMENT_NAME >> $OUTPUT_DIR/last_experiment.txt
  mkdir $OUTPUT_DIR/$EXPERIMENT_NAME
  # run experiment
  cd $EXPERIMENT_DIR/$EXPERIMENT_NAME
  echo "*-* Running training inside $(pwd)"
  accelerate launch --config_file ./gpu_accelerate.yaml run_experiment.py
  cd $ROOT
fi

# adapt model weight path
if $DO_INFERENCE || $DO_EVALUATION; then
  if [[ "$MODEL_WEIGHTS" =~ ^-?[0-9]+$ ]] && (( MODEL_WEIGHTS == -2)); then
    echo "*-* No weight option given. Default model will be used."
    MODEL_WEIGHTS="$RAW_DATA_DIR/../official_best_model/best_segformer3d_brats_performance.pth"
  elif [[ $MODEL_WEIGHTS == "latest" ]]; then
    echo "*-* Latest option for weights will automatically search for latest experiment."
    MODEL_WEIGHTS="$OUTPUT_DIR/$(cat $OUTPUT_DIR/last_experiment.txt)/model_checkpoints/best_dice_checkpoint/pytorch_model.bin"
  else
    MODEL_WEIGHTS="$OUTPUT_DIR/$MODEL_WEIGHTS/model_checkpoints/best_dice_checkpoint/pytorch_model.bin"
  fi
fi

if $DO_INFERENCE; then
  echo "*-* Launching the inference..."
  INFERENCE_CMD="python3 $ROOT/eval_scripts/inference.py"
  
  if [ "$TYPE_INFERENCE" -eq -1 ]; then
    echo "*-* Type 1 inference: on all validation cases."
    INFERENCE_CMD="$INFERENCE_CMD --inf_all --weights $MODEL_WEIGHTS"
  else
    echo "*-* Type 2 inference: on a specific case."
    INFERENCE_CMD="$INFERENCE_CMD --case_number $TYPE_INFERENCE --weights $MODEL_WEIGHTS"
  fi
  
  eval $INFERENCE_CMD
fi

if $DO_EVALUATION; then
  echo "*-* Launching the evaluation..."
  EVALUATION_CMD="python3 $ROOT/eval_scripts/evaluate.py"
  
  if [ "$TYPE_EVALUATION" -eq -1 ]; then
    echo "*-* Type 1 evaluation: on all validation and training cases."
    EVALUATION_CMD="$EVALUATION_CMD --eval_all --weights $MODEL_WEIGHTS"
  else
    echo "*-* Type 2 evaluation: on a specific case."
    EVALUATION_CMD="$EVALUATION_CMD --case_number $TYPE_EVALUATION --weights $MODEL_WEIGHTS"
  fi
  
  eval $EVALUATION_CMD
fi
echo "*-* End of the entrypoint"