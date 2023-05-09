#!/usr/bin/env bash

input=~/ont/data/train/oversample.txt
tmp_file=/tmp/shuf.txt
train=~/ont/data/train/train.txt
test=~/ont/data/train/test.txt

# shuffle
get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}
shuf --random-source=<(get_seeded_random 42) $input -o $tmp_file

# split into train(80%) and test(20%)
total=$(wc -l $tmp_file | awk '{print $1}')
train_len=$((total*80/100))
test_len=$((total-train_len))
echo "total: $total, train_len: $train_len, test_len: $test_len"
head -n $train_len $tmp_file > $train
tail -n $test_len $tmp_file > $test
