#!/bin/bash

# Author: Martin Kostelník
# Brief: Create embeddings of mapping result.
# Date: 24.11.2023

BASE=/home/martin/textbite

source $BASE/../semant/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/baseline
DATA_PATH=$BASE/data/segmentation/full-embeddings-lm72.pkl

python -u $SCRIPTS_DIR/train.py \
    --data $DATA_PATH \
    -b 128 \
    -l 2 \
    -n 1024 \
    -d 0.0 \
    -e 500 \
    --lr 1e-3 \
    --save .
