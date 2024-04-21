#!/bin/bash

# Author: Martin Kostelník
# Brief: Infer using Joiner model
# Date: 03.03.2024

BASE=/home/martin/textbite

source $BASE/../semant/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/joiner
JSON_PATH=$BASE/yoloinference
XML_PATH=$BASE/data/segmentation/xmls/test
SAVE_PATH=$BASE/joinerinference-ae

mkdir -p $SAVE_PATH

python -u $SCRIPTS_DIR/infer_cluster.py \
    --logging-level INFO \
    --data $JSON_PATH \
    --xmls $XML_PATH \
    --save $SAVE_PATH
