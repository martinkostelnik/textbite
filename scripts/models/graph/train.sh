#!/bin/bash

# Author: Martin Kostelník
# Brief: Train graph model
# Date: 09.12.2023

BASE=/home/martin/textbite

source $BASE/../semant/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/graph
DATA_PATH=$BASE/data/segmentation/graph.pkl

python -u $SCRIPTS_DIR/train.py \
    --data $DATA_PATH \
    -e 500 \
    --lr 1e-3
