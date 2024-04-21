#!/bin/bash

# Author: Martin Kostelník
# Brief: Finetune BERT on SGE
# Date: 21.04.2024

BASE=/mnt/matylda1/xkoste12

source $BASE/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/textbite/models/baseline
DATA_PATH=$BASE/textbite-data/nsp-data
SAVE_PATH=$BASE/czerts
FILENAME=data-train.pkl

mkdir -p $SAVE_PATH

python -u $SCRIPTS_DIR/finetune_czert.py \
    --logging-level INFO \
    --data $DATA_PATH \
    --save $SAVE_PATH \
    --lr 1e-3 \
    --epochs 2 \
    --batch-size 32
