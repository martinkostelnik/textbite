#!/bin/bash

# Author: Martin Kostelník
# Brief: Train MLP model for bite joining
# Date: 16.03.2023

BASE=/home/martin/textbite

source $BASE/../semant/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/joiner
GRAPHS_PATH=$BASE/joiner-graphs
DATA_PATH=$GRAPHS_PATH/graphs-train.pkl
DATA_PATH_VAL_BOOK=$GRAPHS_PATH/graphs-val-book.pkl
DATA_PATH_VAL_DICT=$GRAPHS_PATH/graphs-val-dict.pkl
DATA_PATH_VAL_PERI=$GRAPHS_PATH/graphs-val-peri.pkl
SAVE_PATH=$BASE/joiner-models-mlp

mkdir -p $SAVE_PATH

python -u $SCRIPTS_DIR/train_mlp.py \
    --train $DATA_PATH \
    --val-book $DATA_PATH_VAL_BOOK \
    --val-dict $DATA_PATH_VAL_DICT \
    --val-peri $DATA_PATH_VAL_PERI \
    -l 3 \
    -n 64 \
    -d 0.0 \
    --lr 5e-3 \
    --batch-size 64 \
    --epochs 50 \
    --save $SAVE_PATH
