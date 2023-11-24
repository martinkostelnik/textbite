#!/bin/bash

# Author: Martin Kostelník
# Brief: Create embeddings of mapping result.
# Date: 24.11.2023

BASE=/home/martin/textbite

source $BASE/../semant/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/baseline
DATA_PATH=$BASE/novy-nocontext.pkl

python -u $SCRIPTS_DIR/train.py \
    --data $DATA_PATH \
    -b 128 \
    -r 0.9 \
    -l 3 \
    -n 1024 \
    -d 0.0 \
    -e 500 \
    --lr 1e-4 \
    --save $BASE/models
