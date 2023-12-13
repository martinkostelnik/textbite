#!/bin/bash

set -euo pipefail

python -u /mnt/matylda5/ibenes/teh_codez/textbite/textbite/models/graph/train.py \
    --data /mnt/matylda1/xkoste12/textbite-data/graphs-all.pkl \
    --checkpoint-dir /mnt/matylda5/ibenes/projects/textbite/graph-checkpoints \
    --save /mnt/matylda5/ibenes/projects/textbite/tmp-save \
    --layers 2 \
    --hidden-width 256 \
    --output-size 128 \
    --dropout 0.0 \
    --lr 1e-3 \
    --report-interval 1000
