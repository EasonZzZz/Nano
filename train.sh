#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate nano_dev
export PYTHONPATH="/home/eason/test/Nano/"

python3 /home/eason/test/Nano/nano/train.py \
    --train_file ~/ont/data/0318_seq/train_features_over.csv \
    --valid_file ~/ont/data/0318_seq/test_features_over.csv \
    --model_dir ~/ont/data/0318_seq/model_0001_over \
    --num_epochs 10 \
    --lr 0.0001 \
