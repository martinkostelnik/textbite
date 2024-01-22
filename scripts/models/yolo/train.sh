#!/bin/bash

# Author: Martin Kostelník
# Brief: Train graph model for bite joining
# Date: 30.12.2023

BASE=/home/martin/textbite

source $BASE/../semant/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/yolo
DATA_PATH=$BASE/graphs.pkl

python -u $SCRIPTS_DIR/train.py \
    --data $DATA_PATH \
    -l 3 \
    -n 128 \
    -o 128 \
    -d 0.1 \
    --lr 1e-5 \
    --batch-size 256 \
    --report-interval 100 \
    --save $BASE/models \
    --checkpoint-dir $BASE/models
