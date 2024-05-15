#!/bin/bash

# Author: Martin Kostelník
# Brief: Train YOLO model on SGE
# Date: 14.02.2024

BASE=/mnt/matylda1/xkoste12

source $BASE/venv/bin/activate
ulimit -t unlimited

export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=memory.free,index --format=csv,nounits,noheader | sort -nr | head -1 | awk '{ print $NF }')

MODEL_PATH=$BASE/textbite-yolo/yolov8n.pt
DATA_PATH=$BASE/textbite-yolo/data.yaml
EPOCHS=50

mkdir $BASE/yolo-n-640
yolo detect train model=$MODEL_PATH data=$DATA_PATH epochs=$EPOCHS imgsz=640 project=$BASE/yolo-n-640 amp=False workers=4 >$BASE/yolo-n-640/out.txt 2>&1

mkdir $BASE/yolo-n-800
yolo detect train model=$MODEL_PATH data=$DATA_PATH epochs=$EPOCHS imgsz=800 project=$BASE/yolo-n-800 amp=False workers=4 >$BASE/yolo-n-800/out.txt 2>&1

mkdir $BASE/yolo-n-1000
yolo detect train model=$MODEL_PATH data=$DATA_PATH epochs=$EPOCHS imgsz=1000 project=$BASE/yolo-n-1000 amp=False workers=4 >$BASE/yolo-n-1000/out.txt 2>&1

mkdir $BASE/yolo-n-1200
yolo detect train model=$MODEL_PATH data=$DATA_PATH epochs=$EPOCHS imgsz=1200 project=$BASE/yolo-n-1200 amp=False workers=4 >$BASE/yolo-n-1200/out.txt 2>&1

mkdir $BASE/yolo-n-1400
yolo detect train model=$MODEL_PATH data=$DATA_PATH epochs=$EPOCHS imgsz=1400 project=$BASE/yolo-n-1400 amp=False workers=4 >$BASE/yolo-n-1400/out.txt 2>&1
