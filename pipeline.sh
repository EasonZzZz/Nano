#!/usr/bin/env bash

# This script is a pipeline for the analysis of the nanopore sequencing data.
# It takes the multi-fast5 files as input and outputs the re-squiggled single-fast5 files.
# The re-squiggled single-fast5 files are then used for the downstream analysis.

# fast5_dir: The directory containing the multi-fast5 files.
fast5_dir=/home/eason/test/0318_1
# fastq_dir: The directory where the fastq files will be saved.
fastq_dir=/home/eason/test/0318_1/guppy
# ref_fasta: The reference genome in fasta format.
ref_fasta=/home/eason/test/0318_1/ref_960.fasta


# The pipeline is divided into 3 parts:
# 1. Basecalling: The multi-fast5 files are basecalled using the Guppy basecaller.
guppy_basecaller \
  --input_path "$fast5_dir" \
  --save_path "$fastq_dir" \
  --config dna_r9.4.1_450bps_hac.cfg \
  --device cuda:0 --recursive
cat "$fastq_dir"/*/*.fastq > "$fastq_dir"/all.fastq

# 2. Multi-to-single: The multi-fast5 files are converted to single-fast5 files using the ont_fast5_api.
multi_to_single_fast5 \
  --input_path "$fast5_dir" \
  --save_path "$fast5_dir"/single \
  --recursive --threads 12

# 3. Resquiggle: The single-fast5 files are re-squiggled using the Guppy resquiggle.
tombo preprocess annotate_raw_with_fastqs \
  --fast5-basedir "$fast5_dir"/single \
  --fastq-filenames "$fastq_dir"/all.fastq \
  --overwrite --processes 12

tombo resquiggle \
  "$fast5_dir"/single \
  "$ref_fasta" \
  --processes 12 \
  --overwrite