#!/bin/bash

# Author: Martin Kostelník
# Brief: Infer data using YOLO model
# Date: 28.12.2023

BASE=/home/martin/textbite

source $BASE/../semant/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/yolo
IMG_PATH=$BASE/data/segmentation/images/test
XML_PATH=$BASE/data/segmentation/xmls/test
ALTO_PATH=$BASE/data/segmentation/altos
MODEL_PATH=$BASE/yolo-models/yolo-s-1000.pt
SAVE_PATH=$BASE/yoloinference

python -u $SCRIPTS_DIR/infer.py \
    --logging-level INFO \
    --data $XML_PATH \
    --images $IMG_PATH \
    --altos $ALTO_PATH \
    --model $MODEL_PATH \
    --save $SAVE_PATH
