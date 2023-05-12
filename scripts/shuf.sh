#!/usr/bin/env bash

input=~/ont/data/train/oversample.txt
tmp_file=/tmp/shuf.txt
train=~/ont/data/train/train.txt
valid=~/ont/data/train/valid.txt

# shuffle
get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}
shuf --random-source=<(get_seeded_random 42) $input -o $tmp_file

# split into train(80%) and valid(20%)
total=$(wc -l $tmp_file | awk '{print $1}')
train_len=$((total*80/100))
valid_len=$((total-train_len))
echo "total: $total, train_len: $train_len, valid_len: $valid_len"
head -n $train_len $tmp_file > $train
tail -n $valid_len $tmp_file > $valid
rm $tmp_file
