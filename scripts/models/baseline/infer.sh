#!/bin/bash

# Author: Martin Kostelník
# Brief: Infer XML files.
# Date: 04.12.2023

BASE=/home/martin/textbite

source $BASE/../semant/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/baseline
XML_DATA=$BASE/tmp
LANGUAGE_MODEL=$BASE/models/lm72.pth
MODEL=$BASE/models/BaselineModel.pth
OUT_DIR=$BASE/inference-outputs

python -u $SCRIPTS_DIR/infer.py \
    --data $XML_DATA \
    --lm $LANGUAGE_MODEL \
    --model $MODEL \
    --save $OUT_DIR
