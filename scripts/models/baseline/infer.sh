#!/bin/bash

# Author: Martin Kostelník
# Brief: Infer XML files using bert-like language model.
# Date: 17.03.2024

BASE=/home/martin/textbite

source $BASE/venv-old-transformers/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/baseline
XML_PATH=$BASE/data/segmentation/xmls/test
MODEL_PATH=$BASE/models/best-nsp-czert.pth
SAVE_PATH=$BASE/baseline-inference-lm

mkdir -p $SAVE_PATH

python -u $SCRIPTS_DIR/infer.py \
    --logging-level INFO \
    --xmls $XML_PATH \
    --model $MODEL_PATH \
    --threshold 0.5 \
    --method lm \
    --save $SAVE_PATH
