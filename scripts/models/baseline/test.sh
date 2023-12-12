#!/bin/bash

# Author: Martin Kostelník
# Brief: Test baseline model
# Date: 12.12.2023

BASE=/home/martin/textbite

source $BASE/../semant/venv/bin/activate

SCRIPTS_DIR=$BASE/textbite/models/baseline
VIS_SCRIPTS_DIR=$BASE/textbite/visualization

XML_ALL=$BASE/data/segmentation/xmls
XML_BOOKS=$BASE/data/segmentation/test/book
XML_DICTS=$BASE/data/segmentation/test/dictionary
XML_PERIO=$BASE/data/segmentation/test/periodical
IMAGES=$BASE/data/segmentation/images
LANGUAGE_MODEL=$BASE/models/lm72.pth
MODEL=$BASE/models/BaselineModel.pth
OUT_DIR=$BASE/baseline-test-outputs

mkdir $OUT_DIR

echo Infering books ...
mkdir $OUT_DIR/book
python -u $SCRIPTS_DIR/infer.py \
    --data $XML_BOOKS \
    --lm $LANGUAGE_MODEL \
    --model $MODEL \
    --save $OUT_DIR/book

echo Infering dictionaries ...
mkdir $OUT_DIR/dictionary
python -u $SCRIPTS_DIR/infer.py \
    --data $XML_DICTS \
    --lm $LANGUAGE_MODEL \
    --model $MODEL \
    --save $OUT_DIR/dictionary

echo Infering periodicals ...
mkdir $OUT_DIR/periodical
python -u $SCRIPTS_DIR/infer.py \
    --data $XML_PERIO \
    --lm $LANGUAGE_MODEL \
    --model $MODEL \
    --save $OUT_DIR/periodical

echo Visualizing books ...
python -u $VIS_SCRIPTS_DIR/visualize_json.py \
    --images $IMAGES \
    --xml $XML_ALL \
    --jsons $OUT_DIR/book \
    --save $OUT_DIR/book

    
echo Visualizing dictionaries ...
python -u $VIS_SCRIPTS_DIR/visualize_json.py \
    --images $IMAGES \
    --xml $XML_ALL \
    --jsons $OUT_DIR \
    --save $OUT_DIR/dictionary

    
echo Visualizing periodicals ...
python -u $VIS_SCRIPTS_DIR/visualize_json.py \
    --images $IMAGES \
    --xml $XML_ALL \
    --jsons $OUT_DIR \
    --save $OUT_DIR/periodical
