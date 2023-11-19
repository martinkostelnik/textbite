#!/bin/bash

# Author: Martin Kostelník
# Brief: Create embeddings of mapping result.
# Date: 28.10.2023

BASE=/home/martin/textbite

source $BASE/../semant/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/baseline
INPUT_PATH=$BASE/tmp/mapping.txt

python -u $SCRIPTS_DIR/create_embeddings.py \
    -i $INPUT_PATH
