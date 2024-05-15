#!/bin/bash

# Author: Martin Kostelník
# Brief: Evaluate YOLO models
# Date: 28.12.2023

BASE=/home/martin/textbite

source $BASE/../semant/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/yolo
IMG_PATH=$BASE/data/segmentation/images/test
XML_PATH=$BASE/data/segmentation/xmls/test
ALTO_PATH=$BASE/data/segmentation/altos
LABELS_PATH=$BASE/data/segmentation/labels-merged/test
MODEL_DIR=$BASE/yolo-models
SAVE_PATH=$BASE/yolo-evaluation

mkdir $SAVE_PATH
MODELS=$(ls "$MODEL_DIR")

for model in $MODELS
do
    CURRENT_SAVE_PATH=$SAVE_PATH/$model
    echo $CURRENT_SAVE_PATH

    python $SCRIPTS_DIR/infer.py \
        --logging-level INFO \
        --model $MODEL_DIR/$model \
        --data $XML_PATH \
        --altos $ALTO_PATH \
        --images $IMG_PATH \
        --save $CURRENT_SAVE_PATH

    python $SCRIPTS_DIR/../evaluate.py \
        --hypothesis $CURRENT_SAVE_PATH \
        --ground-truth $LABELS_PATH
done
