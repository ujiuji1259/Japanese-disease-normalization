#! /bin/bash

BASE_DIR=$(cd $(dirname $0)/..;pwd)
python $BASE_DIR/similarity/predict_sbert.py --data_path basic --model_path normal --input_path $1 --output_path $2
