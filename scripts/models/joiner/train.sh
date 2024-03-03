#!/bin/bash

# Author: Martin Kostelník
# Brief: Train graph model for bite joining
# Date: 30.12.2023

BASE=/home/martin/textbite

source $BASE/../semant/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/joiner
DATA_PATH=$BASE/joiner-graphs/graphs-train.pkl
DATA_PATH_VAL_BOOK=$BASE/joiner-graphs/graphs-val-book.pkl
DATA_PATH_VAL_DICT=$BASE/joiner-graphs/graphs-val-dict.pkl
DATA_PATH_VAL_PERI=$BASE/joiner-graphs/graphs-val-peri.pkl
SAVE_PATH=$BASE/joinertest

mkdir -p $SAVE_PATH

python -u $SCRIPTS_DIR/train.py \
    --train $DATA_PATH \
    --val-book $DATA_PATH_VAL_BOOK \
    --val-dict $DATA_PATH_VAL_DICT \
    --val-peri $DATA_PATH_VAL_PERI \
    -l 3 \
    -n 128 \
    -o 128 \
    -d 0.3 \
    --threshold 0.5 \
    --lr 1e-5 \
    --batch-size 16 \
    --report-interval 10 \
    --save $SAVE_PATH \
    --checkpoint-dir $SAVE_PATH
