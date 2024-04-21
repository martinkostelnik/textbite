#!/bin/bash

# Author: Martin Kostelník
# Brief: Create NSP dataset on SGE
# Date: 21.04.2024

BASE=/mnt/matylda1/xkoste12

source $BASE/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/textbite/models/baseline
JSON_PATH=$BASE/textbite-data/export-3396-22-01-2024.json
XML_PATH=$BASE/textbite-data/xmls/train
SAVE_PATH=$BASE/nsp-dataset
FILENAME=data-train.pkl

mkdir -p $SAVE_PATH

python -u $SCRIPTS_DIR/create_lm_dataset.py \
    --logging-level INFO \
    --xmls $XML_PATH \
    --tokenizer $BASE/czert \
    --export $JSON_PATH \
    --save $SAVE_PATH \
    --filename $FILENAME

XML_PATH=$BASE/textbite-data/xmls/val
FILENAME=data-val.pkl

python -u $SCRIPTS_DIR/create_lm_dataset.py \
    --logging-level INFO \
    --xmls $XML_PATH \
    --tokenizer $BASE/czert \
    --export $JSON_PATH \
    --save $SAVE_PATH \
    --filename $filename
