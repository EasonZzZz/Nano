#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate nano_dev
export PYTHONPATH="/home/eason/codes/Nano/"

data_dir="/home/eason/nvme/ont/data"
ref_dir="/home/eason/nvme/ont/ref"
data_names=("0918_seq/0918_seq1" "0918_seq/0918_seq2" "0921_seq/0921_seq_r" "0927_seq/0927_seq1" "0927_seq/0927_seq2" "0927_seq/0927_seq3")
ref_names=("0918" "0918" "0921" "0927" "0927" "0927")
output_file=$data_dir/train/oversample.txt

total_start_time=$(date +%s)
for ((i=0; i<${#data_names[@]}; i++))
do
  echo "########## ${data_names[$i]} ##########"
  echo "extract ${ref_names[$i]} features from ${data_names[$i]}"
  start_time=$(date +%s)
#  python /home/eason/codes/Nano/nano/extract_features.py \
#    --fast5_dir $data_dir/"${data_names[$i]}"/960_subset/single \
#    --output_file $data_dir/"${data_names[$i]}"/960_subset/unmethyl.txt \
#    --overwrite -p 24 --positions ~/ont/data/train/25_barcode_unmethyl.csv \
#    --reference $ref_dir/"${ref_names[$i]}".fa --methyl_label 0
#
#  python /home/eason/codes/Nano/nano/extract_features.py \
#    --fast5_dir $data_dir/"${data_names[$i]}"/960_subset/single \
#    --output_file $data_dir/"${data_names[$i]}"/960_subset/methyl.txt \
#    --overwrite -p 24 --positions ~/ont/data/train/25_barcode_methyl.csv \
#    --reference $ref_dir/"${ref_names[$i]}".fa --methyl_label 1

  echo "Simple Oversampling"
  methyl_len=$(wc -l $data_dir/"${data_names[$i]}"/960_subset/methyl.txt | awk '{print $1}')
  unmethyl_len=$(wc -l $data_dir/"${data_names[$i]}"/960_subset/unmethyl.txt | awk '{print $1}')
  times=$((unmethyl_len/methyl_len))
  echo "methyl_len: $methyl_len, unmethyl_len: $unmethyl_len, times: $times"
  for ((j=0; j<times; j++))
  do
    cat $data_dir/"${data_names[$i]}"/960_subset/methyl.txt >> $output_file
  done
  cat $data_dir/"${data_names[$i]}"/960_subset/unmethyl.txt >> $output_file

  end_time=$(date +%s)
  echo "extract ${ref_names[$i]} features from ${data_names[$i]} done, time: $((end_time-start_time))s"
  echo "########## ${data_names[$i]} ##########"
  echo ""
done
total_end_time=$(date +%s)
echo "total time: $((total_end_time-total_start_time))s"