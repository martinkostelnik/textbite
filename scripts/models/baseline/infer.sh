#!/bin/bash

# Author: Martin Kostelník
# Brief: Infer XML files using bert-like language model.
# Date: 17.03.2024

BASE=/home/martin/textbite

source $BASE/venv-old-transformers/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/baseline
XML_PATH=$BASE/data/segmentation/xmls/test
MODEL_PATH=$BASE/models/lm264.pth
SAVE_PATH=$BASE/baseline-inference

mkdir -p $SAVE_PATH

python -u $SCRIPTS_DIR/infer.py \
    --logging-level INFO \
    --xmls $XML_PATH \
    --model $MODEL_PATH \
    --threshold 0.005 \
    --method dist \
    --save $SAVE_PATH
