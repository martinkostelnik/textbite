#!/bin/bash

python -u /mnt/matylda5/ibenes/teh_codez/textbite/textbite/models/graph/train.py \
    --data /mnt/matylda1/xkoste12/textbite-data/graphs.pkl \
    -e 500 \
    --lr 3e-3
