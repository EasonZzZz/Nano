#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate nano_dev
export PYTHONPATH="/home/eason/codes/Nano/"

#echo "preprocessing 0921_seq_r"
#./pipeline.sh ~/nvme/ont/data/0921_seq/0921_seq_r/subset/ \
#    ~/nvme/ont/ref/0921.fa \
#    ~/nvme/ont/data/0921_seq/0921_seq_r/960_subset/ \
#    24

echo "extract_features from 0918_seq"
python /home/eason/codes/Nano/nano/extract_features.py \
    --fast5_dir ~/nvme/ont/data/0918_seq/0918_seq1/960_subset/single \
    --output_file ~/nvme/ont/data/0918_seq/0918_seq1/960_subset/unmethyl.txt \
    --overwrite -p 24 --positions ~/ont/data/train/25_barcode_unmethyl.csv \
    --reference ~/nvme/ont/ref/0918.fa --methyl_label 0

python /home/eason/codes/Nano/nano/extract_features.py \
    --fast5_dir ~/nvme/ont/data/0918_seq/0918_seq1/960_subset/single \
    --output_file ~/nvme/ont/data/0918_seq/0918_seq1/960_subset/methyl.txt \
    --overwrite -p 24 --positions ~/ont/data/train/25_barcode_methyl.csv \
    --reference ~/nvme/ont/ref/0918.fa --methyl_label 1


echo "extract_features from 0918_seq"
python /home/eason/codes/Nano/nano/extract_features.py \
    --fast5_dir ~/nvme/ont/data/0918_seq/0918_seq2/960_subset/single \
    --output_file ~/nvme/ont/data/0918_seq/0918_seq2/960_subset/unmethyl.txt \
    --overwrite -p 24 --positions ~/ont/data/train/25_barcode_unmethyl.csv \
    --reference ~/nvme/ont/ref/0918.fa --methyl_label 0

python /home/eason/codes/Nano/nano/extract_features.py \
    --fast5_dir ~/nvme/ont/data/0918_seq/0918_seq2/960_subset/single \
    --output_file ~/nvme/ont/data/0918_seq/0918_seq2/960_subset/methyl.txt \
    --overwrite -p 24 --positions ~/ont/data/train/25_barcode_methyl.csv \
    --reference ~/nvme/ont/ref/0918.fa --methyl_label 1


echo "extract_features from 0921_seq_r"
python /home/eason/codes/Nano/nano/extract_features.py \
    --fast5_dir ~/nvme/ont/data/0921_seq/0921_seq_r/960_subset/single \
    --output_file ~/nvme/ont/data/0921_seq/0921_seq_r/960_subset/unmethyl.txt \
    --overwrite -p 24 --positions ~/ont/data/train/25_barcode_unmethyl.csv \
    --reference ~/nvme/ont/ref/0921.fa --methyl_label 0

python /home/eason/codes/Nano/nano/extract_features.py \
    --fast5_dir ~/nvme/ont/data/0921_seq/0921_seq_r/960_subset/single \
    --output_file ~/nvme/ont/data/0921_seq/0921_seq_r/960_subset/methyl.txt \
    --overwrite -p 24 --positions ~/ont/data/train/25_barcode_methyl.csv \
    --reference ~/nvme/ont/ref/0921.fa --methyl_label 1


echo "extract_features from 0927_seq1"
python /home/eason/codes/Nano/nano/extract_features.py \
    --fast5_dir ~/nvme/ont/data/0927_seq/0927_seq1/960_subset/single \
    --output_file ~/nvme/ont/data/0927_seq/0927_seq1/960_subset/unmethyl.txt \
    --overwrite -p 24 --positions ~/ont/data/train/25_barcode_unmethyl.csv \
    --reference ~/nvme/ont/ref/0927_3.fa --methyl_label 0

python /home/eason/codes/Nano/nano/extract_features.py \
    --fast5_dir ~/nvme/ont/data/0927_seq/0927_seq1/960_subset/single \
    --output_file ~/nvme/ont/data/0927_seq/0927_seq1/960_subset/methyl.txt \
    --overwrite -p 24 --positions ~/ont/data/train/25_barcode_methyl.csv \
    --reference ~/nvme/ont/ref/0927_3.fa --methyl_label 1


echo "extract_features from 0927_seq2"
python /home/eason/codes/Nano/nano/extract_features.py \
    --fast5_dir ~/nvme/ont/data/0927_seq/0927_seq2/960_subset/single \
    --output_file ~/nvme/ont/data/0927_seq/0927_seq2/960_subset/unmethyl.txt \
    --overwrite -p 24 --positions ~/ont/data/train/25_barcode_unmethyl.csv \
    --reference ~/nvme/ont/ref/0927_3.fa --methyl_label 0

python /home/eason/codes/Nano/nano/extract_features.py \
    --fast5_dir ~/nvme/ont/data/0927_seq/0927_seq2/960_subset/single \
    --output_file ~/nvme/ont/data/0927_seq/0927_seq2/960_subset/methyl.txt \
    --overwrite -p 24 --positions ~/ont/data/train/25_barcode_methyl.csv \
    --reference ~/nvme/ont/ref/0927_3.fa --methyl_label 1


echo "extract_features from 0927_seq3"
python /home/eason/codes/Nano/nano/extract_features.py \
    --fast5_dir ~/nvme/ont/data/0927_seq/0927_seq3/960_subset/single \
    --output_file ~/nvme/ont/data/0927_seq/0927_seq3/960_subset/unmethyl.txt \
    --overwrite -p 24 --positions ~/ont/data/train/25_barcode_unmethyl.csv \
    --reference ~/nvme/ont/ref/0927_3.fa --methyl_label 0

python /home/eason/codes/Nano/nano/extract_features.py \
    --fast5_dir ~/nvme/ont/data/0927_seq/0927_seq3/960_subset/single \
    --output_file ~/nvme/ont/data/0927_seq/0927_seq3/960_subset/methyl.txt \
    --overwrite -p 24 --positions ~/ont/data/train/25_barcode_methyl.csv \
    --reference ~/nvme/ont/ref/0927_3.fa --methyl_label 1
