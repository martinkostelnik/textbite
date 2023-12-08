#!/bin/bash

# Author: Martin Kostelník
# Brief: Create pickle containing graphs
# Date: 09.12.2023

BASE=/home/martin/textbite

source $BASE/../semant/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/graph
EMBEDDINGS_PATH=$BASE/data/segmentation/graph-embeddings.pkl
XML_PATH=$BASE/tmp/

python -u $SCRIPTS_DIR/create_graphs.py \
    --xml $XML_PATH \
    --embeddings $EMBEDDINGS_PATH
