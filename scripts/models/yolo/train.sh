#!/bin/bash

# Author: Martin Kostelník
# Brief: Train YOLO model on SGE
# Date: 31.01.2024

BASE=/mnt/matylda1/xkoste12

source $BASE/venv/bin/activate
ulimit -t unlimited

export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')

MODEL_PATH=$BASE/textbite-yolo/yolov8n.pt
DATA_PATH=$BASE/textbite-yolo/data.yaml
SAVE_PATH=$BASE/yolo-n-640
EPOCHS=10

mkdir $SAVE_PATH

yolo detect train model=$MODEL_PATH data=$DATA_PATH epochs=$EPOCHS imgsz=640 project=$SAVE_PATH amp=False >$SAVE_PATH/out.txt 2>&1
