#!/bin/bash

# Author: Martin Kostelník
# Brief: Benchmark inference time of yolo models
# Date: 17.04.2024

BASE=/home/martin/textbite

source $BASE/../semant/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/yolo
IMG_PATH=$BASE/data/segmentation/images/test
MODEL_DIR=$BASE/yolo/yolo-models-200

MODELS=$(ls "$MODEL_DIR")

for model in $MODELS
do
    nocache python -u $SCRIPTS_DIR/benchmark_inference_time.py \
        --logging-level INFO \
        --images $IMG_PATH \
        --model $MODEL_DIR/$model
done
