#!/bin/bash

# Author: Martin Kostelník
# Brief: Create mapping between label-studio export and PERO-OCR XMLs.
# Date: 27.10.2023

BASE=/home/martin/textbite

source $BASE/../semant/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/label_studio
JSON_PATH=$BASE/data/dataset/data.json
XML_PATH=$BASE/tmp

python -u $SCRIPTS_DIR/mapping.py \
    --json $JSON_PATH \
    --xml $XML_PATH