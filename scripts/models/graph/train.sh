#!/bin/bash

# Author: Martin Kostelník
# Brief: Train graph model
# Date: 09.12.2023

BASE=/home/martin/textbite

source $BASE/../semant/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/graph
DATA_PATH=$BASE/data/segmentation/graphs-all.pkl

python -u $SCRIPTS_DIR/train.py \
    --data $DATA_PATH \
    -l 3 \
    -n 64 \
    -o 64 \
    -d 0.0 \
    --lr 1e-2 \
    --batch-size 64 \
    --report-interval 100 \
    --save $BASE/models \
    --checkpoint-dir $BASE/models
