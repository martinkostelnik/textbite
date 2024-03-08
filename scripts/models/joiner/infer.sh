#!/bin/bash

# Author: Martin Kostelník
# Brief: Infer using Joiner model
# Date: 03.03.2024

BASE=/home/martin/textbite

source $BASE/../semant/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/joiner
XML_PATH=$BASE/data/segmentation/xmls/test
IMG_PATH=$BASE/data/segmentation/images/test
YOLO_PATH=$BASE/yolo-models-20-02-24/yolo-s-1000.pt
MODEL_PATH=$BASE/joiner-models/JoinerGraphModel-joiner-checkpoint.159.pth
NORMALIZER_PATH=$BASE/joiner-models/normalizer.pkl
SAVE_PATH=$BASE/joinerinference

mkdir -p $SAVE_PATH

python -u $SCRIPTS_DIR/infer.py \
    --logging-level INFO \
    --xmls $XML_PATH \
    --images $IMG_PATH \
    --yolo $YOLO_PATH \
    --model $MODEL_PATH \
    --normalizer $NORMALIZER_PATH \
    --save $SAVE_PATH
