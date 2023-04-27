#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate nano_dev
export PYTHONPATH="/home/eason/test/Nano/"

python /home/eason/test/Nano/nano/extract_features.py \
    --fast5_dir ~/ont/data/0420_seq/barcode_2/no_sample \
    --output_dir ~/ont/data/0420_seq/barcode_2/960/features \
    --overwrite -p 12 -obs 100000 \
    --reference ~/ont/ref/ref_960.fasta

#./pipeline.sh ~/ont/data/0921_seq/0921_seq_r/no_sample/ ~/ont/ref/ref_960.fasta ~/ont/data/0921_seq/0921_seq_r/960 24
python /home/eason/test/Nano/nano/extract_features.py \
    --fast5_dir ~/ont/data/0420_seq/barcode_3/no_sample \
    --output_dir ~/ont/data/0420_seq/barcode_3/960/features \
    --overwrite -p 12 -obs 100000 \
    --reference ~/ont/ref/ref_960.fasta
