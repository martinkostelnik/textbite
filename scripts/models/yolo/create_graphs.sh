#!/bin/bash

# Author: Martin Kostelník
# Brief: Create graphs for bit joining
# Date: 30.12.2023

BASE=/home/martin/textbite

source $BASE/../semant/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/yolo
IMG_PATH=$BASE/data/segmentation/backup/images
XML_PATH=$BASE/data/segmentation/backup/xmls
JSON_PATH=$BASE/data/segmentation/export-1828-12-12-2023.json
MODEL_PATH=$BASE/models/yolov8s.pt
SAVE_PATH=$BASE

python -u $SCRIPTS_DIR/create_graphs.py \
    --logging-level ERROR \
    --xmls $XML_PATH \
    --images $IMG_PATH \
    --json $JSON_PATH \
    --model $MODEL_PATH \
    --save $SAVE_PATH
