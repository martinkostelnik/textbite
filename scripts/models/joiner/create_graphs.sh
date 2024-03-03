#!/bin/bash

# Author: Martin Kostelník
# Brief: Create graphs for GNN joiner based model
# Date: 01.02.2024

BASE=/home/martin/textbite

source $BASE/../semant/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/joiner
JSON_PATH=$BASE/data/segmentation/export-3396-22-01-2024.json
MODEL_PATH=$BASE/yolo-models/yolo-s-800.pt
IMG_PATH=$BASE/data/segmentation/images/train
XML_PATH=$BASE/data/segmentation/xmls/train
SAVE_PATH=$BASE/joiner-graphs-ones
FILENAME=graphs-train.pkl

mkdir -p $SAVE_PATH

python -u $SCRIPTS_DIR/create_graphs.py \
    --logging-level INFO \
    --json $JSON_PATH \
    --model $MODEL_PATH \
    --images $IMG_PATH \
    --xmls $XML_PATH \
    --save $SAVE_PATH \
    --filename $FILENAME

IMG_PATH=$BASE/data/segmentation/images/val-book
XML_PATH=$BASE/data/segmentation/xmls/val-book
FILENAME=graphs-val-book.pkl

python -u $SCRIPTS_DIR/create_graphs.py \
    --logging-level INFO \
    --json $JSON_PATH \
    --model $MODEL_PATH \
    --images $IMG_PATH \
    --xmls $XML_PATH \
    --save $SAVE_PATH \
    --filename $FILENAME

IMG_PATH=$BASE/data/segmentation/images/val-dict
XML_PATH=$BASE/data/segmentation/xmls/val-dict
FILENAME=graphs-val-dict.pkl

python -u $SCRIPTS_DIR/create_graphs.py \
    --logging-level INFO \
    --json $JSON_PATH \
    --model $MODEL_PATH \
    --images $IMG_PATH \
    --xmls $XML_PATH \
    --save $SAVE_PATH \
    --filename $FILENAME

IMG_PATH=$BASE/data/segmentation/images/val-peri
XML_PATH=$BASE/data/segmentation/xmls/val-peri
FILENAME=graphs-val-peri.pkl

python -u $SCRIPTS_DIR/create_graphs.py \
    --logging-level INFO \
    --json $JSON_PATH \
    --model $MODEL_PATH \
    --images $IMG_PATH \
    --xmls $XML_PATH \
    --save $SAVE_PATH \
    --filename $FILENAME
