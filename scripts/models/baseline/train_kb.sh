#!/bin/bash

# Author: Martin Kostelník
# Brief: Create embeddings of mapping result.
# Date: 24.11.2023

DATA_PATH=/mnt/matylda1/xkoste12/textbite-data/data_lrcontext.pkl

python -u /mnt/matylda5/ibenes/teh_codez/textbite/textbite/models/baseline/train.py \
    --data "$DATA_PATH" \
    --batch-size 128 \
    --train-ratio 0.8 \
    --nb-hidden 2 \
    --hidden-width 1024 \
    --dropout 0.0 \
    --max-epochs 500 \
    --lr 1e-3 \
    --save /mnt/matylda5/ibenes/projects/textbite
