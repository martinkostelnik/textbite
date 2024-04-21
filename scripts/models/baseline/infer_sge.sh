#!/bin/bash

# Author: Martin Kostelník
# Brief: Infer baseline on sge
# Date: 21.04.2024

BASE=/mnt/matylda1/xkoste12

source $BASE/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/textbite/models/baseline
XML_PATH=$BASE/textbite-data/xmls/test
MODEL_PATH=$BASE/czerts/nsp-best-czert.pth
SAVE_PATH=$BASE/baseline-inference-lm

mkdir -p $SAVE_PATH

python -u $SCRIPTS_DIR/infer.py \
    --logging-level INFO \
    --xmls $XML_PATH \
    --model $MODEL_PATH \
    --threshold 0.5 \
    --method lm \
    --save $SAVE_PATH