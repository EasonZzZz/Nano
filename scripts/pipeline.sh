#!/usr/bin/env bash

# This script is a pipeline for the analysis of the nanopore sequencing data.
# It takes the multi-data files as input and outputs the re-squiggled single-data files.
# The re-squiggled single-data files are then used for the downstream analysis.

# Environment: The pipeline is tested on Ubuntu 20.04 LTS.
# The following softwares are required:
# 1. Guppy: https://community.nanoporetech.com/downloads
# 2. ont_fast5_api: https://github.com/nanoporetech/ont_fast5_api
# 3. Tombo: https://github.com/nanoporetech/tombo
# 4. vbz_compression: https://github.com/nanoporetech/vbz_compression
# 5. minimap2: https://github.com/lh3/minimap2

eval "$(conda shell.bash hook)"
conda activate nano_dev

case $# in
  4) ;;
  *) echo "Usage: $0 <base_dir> <ref_fasta> <output_dir> <processes>" >&2
     exit 1 ;;
esac

if [ ! -d "$1" ]; then
  echo "Error: $1 is not a directory" >&2
  exit 1
fi

case "$2" in
  *.fa|*.fasta) ;;
  *) echo "Error: $2 is not a fasta file" >&2
     exit 1 ;;
esac

if [ ! -d "$3" ]; then
  rm -rf "$3"
  mkdir "$3"
fi

if [ "$4" -lt 1 ]; then
  echo "Error: $4 is not a positive integer" >&2
  exit 1
fi

# base_dir: The base directory of the data.
base_dir=$1
# ref_fasta: The reference genome in fasta format.
ref_fasta=$2
# output_dir: The output directory.
output_dir=$3
# procs: The number of threads.
procs=$4



# The pipeline is divided into 3 parts:
# 1. Multi-to-single: The multi-data files are converted to single-data files using the ont_fast5_api.
echo "################################"
echo "########### Pipeline ###########"
echo "################################"
echo "######## Multi-to-single #######"
multi_to_single_fast5 \
 --input_path "$base_dir" \
 --save_path "$output_dir"/single \
 --recursive --threads "$procs"

# 2. Basecalling: The multi-data files are basecalled using the Guppy basecaller.
echo "########## Basecalling ##########"
guppy_basecaller \
  --input_path "$output_dir"/single \
  --save_path "$output_dir"/guppy \
  --config dna_r9.4.1_450bps_hac.cfg \
  --device cuda:0 --recursive
cat "$output_dir"/guppy/*/*.fastq > "$output_dir"/guppy/all.fastq

# 3. Annotate and resquiggle: The single-data files are annotated and re-squiggled using the Tombo.
echo "########## Annotate ##########"
tombo preprocess annotate_raw_with_fastqs \
  --fast5-basedir "$output_dir"/single \
  --fastq-filenames "$output_dir"/guppy/all.fastq \
  --sequencing-summary-filenames "$output_dir"/guppy/sequencing_summary.txt \
  --overwrite --processes "$procs"

echo "########## Resquiggle ##########"
tombo resquiggle \
  "$output_dir"/single \
  "$ref_fasta" \
  --processes "$procs" \
  --overwrite