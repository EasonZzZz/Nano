#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate nano_dev
export PYTHONPATH="/home/eason/codes/Nano/"

train=~/ont/data/train/kmer7/train.txt
test=~/ont/data/train/kmer7/valid.txt
model_dir=~/ont/data/train/model/model_001_kmer7

python3 /home/eason/codes/Nano/nano/train.py \
    --train_file $train --valid_file $test \
    --model_dir $model_dir \
    --num_epochs 5 --lr 0.001 \
    --batch_size 16384 \
    --kmer_len 7
