#!/bin/bash

# Author: Martin Kostelník
# Brief: Create embeddings.
# Date: 04.12.2023

BASE=/home/martin/textbite

source $BASE/../semant/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/baseline
XML_DATA=$BASE/tmp
LABELS_FILE=$BASE/data/segmentation/ids-to-labels.txt
LANGUAGE_MODEL=$BASE/models/lm72.pth
OUT_DIR=$BASE/.

python -u $SCRIPTS_DIR/embedding.py \
    --xml $XML_DATA \
    --mapping $LABELS_FILE \
    --model $LANGUAGE_MODEL \
    --save $OUT_DIR
