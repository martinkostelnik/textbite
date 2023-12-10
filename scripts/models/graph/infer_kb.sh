#!/bin/bash

set -euo pipefail

python -u /mnt/matylda5/ibenes/teh_codez/textbite/textbite/models/graph/infer.py \
    --model "$1" \
    --data "/mnt/matylda1/xkoste12/textbite-data/graphs.pkl" \
    --normalizer "/mnt/matylda5/ibenes/projects/textbite/graph-normalizer.pkl"
