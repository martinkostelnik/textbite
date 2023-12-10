#!/bin/bash

set -euo pipefail

python -u /mnt/matylda5/ibenes/teh_codez/textbite/textbite/models/graph/train.py \
    --data /mnt/matylda1/xkoste12/textbite-data/graphs.pkl \
    --checkpoint-dir /mnt/matylda5/ibenes/projects/textbite/graph-checkpoints \
    --normalizer /mnt/matylda5/ibenes/projects/textbite/graph-normalizer.pkl \
    --lr 3e-4 \
    --report-interval 10
