#!/bin/bash

# Author: Martin Kostelník
# Brief: Find best coefficient for dist scaling
# Date: 18.04.2024

BASE=/home/martin/textbite

source $BASE/../semant/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/baseline
XML_PATH=$BASE/data/segmentation/xmls/test
SAVE_PATH=$BASE/baseline-inference-dist

for ((i=20; i<=1000; i+=5)); do
    threshold=$(bc <<< "scale=2; $i/100")
    echo $threshold

    mkdir -p $SAVE_PATH

    python -u $SCRIPTS_DIR/infer.py \
        --logging-level ERROR \
        --xmls $XML_PATH \
        --threshold $threshold \
        --method dist \
        --save $SAVE_PATH

    python -u $SCRIPTS_DIR/../evaluate.py \
        --ground-truth $BASE/data/segmentation/labels-merged/test \
        --hypothesis $SAVE_PATH \
        --logging-level ERROR

    rm -rf $SAVE_PATH
done
