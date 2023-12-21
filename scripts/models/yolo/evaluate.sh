#!/bin/bash

set -euo pipefail

usage="usage: $0 model_path out_dir"

if [ "$#" -ne 2 ]
then
    echo "$usage" >&2
    exit 2
fi

model_path=$1
out_dir=$2
mkdir -p "$out_dir"

xml_dir=/mnt/matylda1/xkoste12/textbite-data/xmls
img_dir=/mnt/matylda1/xkoste12/textbite-data/images
label_dir=/mnt/matylda1/xkoste12/textbite-data/labels-merged

portions=$(ls "$label_dir" | grep -v train)

# the following better gets replaced by proper installation later down the road
TEXTBITE_ROOT=/mnt/matylda5/ibenes/teh_codez/textbite

for portion in $portions
do
    python3 $TEXTBITE_ROOT/textbite/models/yolo/infer.py \
        --model "$model_path" \
        --data "$xml_dir/$portion" \
        --images "$img_dir/$portion" \
        --save "$out_dir/$portion" \
        --logging-level INFO \
        2> "$out_dir/$portion.infer.log"

    python3 $TEXTBITE_ROOT/textbite/models/evaluate.py \
        --hypothesis "$out_dir/$portion" \
        --ground-truth "$label_dir/$portion" \
        2> "$out_dir/$portion.evaluate.log" |\
        tee "$out_dir/$portion.results"
done
