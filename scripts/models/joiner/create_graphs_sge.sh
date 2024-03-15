#!/bin/bash

# Author: Martin Kostelník
# Brief: Create graphs for GNN joiner based model on SGE
# Date: 15.03.2024

BASE=/mnt/matylda1/xkoste12

source $BASE/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/textbite/models/joiner
JSON_PATH=$BASE/textbite/data/segmentation/export-3396-22-01-2024.json
MODEL_PATH=$BASE/textbite-yolo/yolo-s-800.pt
IMG_PATH=$BASE/textbite/data/segmentation/images/train
XML_PATH=$BASE/textbite/data/segmentation/xmls/train
SAVE_PATH=$BASE/joiner-graphs
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

IMG_PATH=$BASE/textbite/data/segmentation/images/val-book
XML_PATH=$BASE/textbite/data/segmentation/xmls/val-book
FILENAME=graphs-val-book.pkl

python -u $SCRIPTS_DIR/create_graphs.py \
    --logging-level INFO \
    --json $JSON_PATH \
    --model $MODEL_PATH \
    --images $IMG_PATH \
    --xmls $XML_PATH \
    --save $SAVE_PATH \
    --filename $FILENAME

IMG_PATH=$BASE/textbite/data/segmentation/images/val-dict
XML_PATH=$BASE/textbite/data/segmentation/xmls/val-dict
FILENAME=graphs-val-dict.pkl

python -u $SCRIPTS_DIR/create_graphs.py \
    --logging-level INFO \
    --json $JSON_PATH \
    --model $MODEL_PATH \
    --images $IMG_PATH \
    --xmls $XML_PATH \
    --save $SAVE_PATH \
    --filename $FILENAME

IMG_PATH=$BASE/textbite/data/segmentation/images/val-peri
XML_PATH=$BASE/textbite/data/segmentation/xmls/val-peri
FILENAME=graphs-val-peri.pkl

python -u $SCRIPTS_DIR/create_graphs.py \
    --logging-level INFO \
    --json $JSON_PATH \
    --model $MODEL_PATH \
    --images $IMG_PATH \
    --xmls $XML_PATH \
    --save $SAVE_PATH \
    --filename $FILENAME
