#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate nano_dev
export PYTHONPATH="/home/eason/codes/Nano/"

train=~/ont/data/train/train.txt
test=~/ont/data/train/test.txt
model_dir=~/ont/data/train/model_0001

start_time=$(date +%s)

python3 /home/eason/codes/Nano/nano/train.py \
    --train_file $train --valid_file $test \
    --model_dir $model_dir \
    --num_epochs 10 --lr 0.0001 --batch_size 2048

end_time=$(date +%s)
echo "total time: $((end_time-start_time))s"
