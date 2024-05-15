#!/bin/bash

# Author: Martin Kostelník
# Brief: Infer using Joiner model
# Date: 03.03.2024

BASE=/home/martin/textbite

source $BASE/../semant/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/joiner
JSON_PATH=$BASE/inference/yolo-inference
XML_PATH=$BASE/data/segmentation/xmls/test
MODEL_PATH=$BASE/joiner-model-final/best-gcn-joiner.pth
NORMALIZER_PATH=$BASE/joiner-model-final/normalizer.pkl
SAVE_PATH=$BASE/joiner-inference

mkdir -p $SAVE_PATH

python -u $SCRIPTS_DIR/infer.py \
    --logging-level INFO \
    --data $JSON_PATH \
    --xmls $XML_PATH \
    --model $MODEL_PATH \
    --normalizer $NORMALIZER_PATH \
    --threshold 0.68 \
    --save $SAVE_PATH
