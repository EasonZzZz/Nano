"""
this file is used to extract features from the dataset.
"""

import os
import sys
import argparse
import time

import h5py
import numpy as np
import pandas as pd
import multiprocessing as mp

from statsmodels.robust import mad
from statsmodels.stats.stattools import robust_kurtosis, robust_skewness
from nano.utils.process_utils import Queue, get_fast5s, get_motif_seqs, get_ref_loc_of_methyl_site
from nano.utils.ref_helper import DNAReference
from nano.utils import logging

reads_group = "Raw/Reads"
global_key = "UniqueGlobalKey"
queue_size_border = 1000
sleep_time = 1
logger = logging.get_logger("extract_features")


def _rescale_signals(raw_signal, scaling, offset):
    """
    Rescale the raw signal.
    """
    return np.array(scaling * (raw_signal + offset), dtype=np.float)


def get_raw_signal(fast5_file, corrected_group, basecall_subgroup):
    """
    Get the raw signal from a data file.
    """
    try:
        fast5_data = h5py.File(fast5_file, "r")
    except IOError:
        raise IOError("Could not open data file: {}".format(fast5_file))

    try:
        align_status = fast5_data["Analyses"][corrected_group][basecall_subgroup].attrs["status"]
    except Exception:
        raise RuntimeError("Could not find status under Analyses/{}/{}/"
                           " in data file: {}".format(corrected_group, basecall_subgroup, fast5_file))
    if align_status != "success":
        print("Alignment failed for data file: {}".format(fast5_file), file=sys.stderr)
        return None, None, None

    info = {}
    try:
        raw_signal = list(fast5_data[reads_group].values())[0]
        info["read_id"] = raw_signal.attrs["read_id"].decode("utf-8")
        raw_signal = raw_signal["Signal"][()]
    except Exception:
        raise RuntimeError("Could not find raw signal under Raw/Reads/Read_[read#]"
                           " in data file: {}".format(fast5_file))

    try:
        channel_info = dict(list(fast5_data[global_key]["channel_id"].attrs.items()))
        scaling = channel_info["range"] / channel_info["digitisation"]
        offset = channel_info["offset"]
        raw_signal = _rescale_signals(raw_signal, scaling, offset)
    except Exception:
        raise RuntimeError("Could not find channel info under UniqueGlobalKey/channel_id"
                           " in data file: {}".format(fast5_file))

    try:
        alignment = dict(list(fast5_data["Analyses"][corrected_group][basecall_subgroup]["Alignment"].attrs.items()))
        info["strand"] = alignment["mapped_strand"]
        info["chrom"] = alignment["mapped_chrom"]
        info["chrom_start"] = alignment["mapped_start"]
        info["chrom_end"] = alignment["mapped_end"]
    except Exception:
        raise RuntimeError("Could not find strand under Analyses/{}/{}/Alignment"
                           " in data file: {}".format(corrected_group, basecall_subgroup, fast5_file))

    try:
        events = fast5_data["Analyses"][corrected_group][basecall_subgroup]["Events"]
    except Exception:
        raise RuntimeError("Could not find events under Analyses/{}/{}/Events"
                           " in data file: {}".format(corrected_group, basecall_subgroup, fast5_file))

    try:
        events_attrs = dict(list(events.attrs.items()))
        read_start_rel_to_raw = events_attrs["read_start_rel_to_raw"]
        starts = list(map(lambda x: x + read_start_rel_to_raw, events["start"]))
    except Exception:
        raise RuntimeError("Could not find read_start_rel_to_raw attribute under Analyses/{}/{}/Events"
                           " in data file: {}".format(corrected_group, basecall_subgroup, fast5_file))

    lengths = events["length"].astype(np.int)
    base = [x.decode("utf-8") for x in events["base"]]
    assert len(starts) == len(lengths) == len(base)
    events = pd.DataFrame({"start": starts, "length": lengths, "base": base})
    return raw_signal, events, info


def normalize_signal(raw_signal):
    """
    Normalize the raw signal.
    """
    # try zsore
    # return (raw_signal - np.mean(raw_signal)) / np.std(raw_signal)

    # try mad
    return (raw_signal - np.median(raw_signal)) / mad(raw_signal)


def _fill_fast5s_queue(fast5s_queue, fast5s, batch_size):
    """
    Fill the fast5s queue.
    """
    for i in np.arrange(0, len(fast5s), batch_size):
        fast5s_queue.put(fast5s[i: i + batch_size])
    return fast5s_queue


def _preprocess(fast5_dir, recursive, motifs, batch_size):
    """
    Preprocess the data files.
    """
    fast5s = get_fast5s(fast5_dir, recursive)
    if len(fast5s) == 0:
        raise RuntimeError("No data files found in directory: {}".format(fast5_dir))
    print("Found {} data files.".format(len(fast5s)))
    fast5s_queue = _fill_fast5s_queue(Queue(), fast5s, batch_size)

    motif_seqs = get_motif_seqs(motifs)

    return fast5s_queue, motif_seqs, len(fast5s)


def _extract_features(
        fast5s, corrected_group, basecall_subgroup, ref,
        motif_seqs, mod_loc, kmers, methyl_label
):
    """
    Extract features from data files.
    todo 抽取特征
    """
    num_bases = (kmers - 1) // 2

    features_list = []
    for fast5 in fast5s:
        raw_signal, events, info = get_raw_signal(fast5, corrected_group, basecall_subgroup)
        if raw_signal is None:
            continue

        raw_signal = normalize_signal(raw_signal)
        seq, signal_list = "", []
        for _, row in events.iterrows():
            seq += row["base"]
            signal_list.append(raw_signal[row["start"]: row["start"] + row["length"]])

        strand, chrom, chrom_start, chrom_end = info["strand"], info["chrom"], info["chrom_start"], info["chrom_end"]
        try:
            chrom_len = ref.get_reference_length(chrom)
        except KeyError:
            logger.warning("Chromosome {} not found in reference.".format(chrom))
            chrom_len = 0

        mod_sites = get_ref_loc_of_methyl_site(seq, motif_seqs, mod_loc)
        for mod_loc_in_read in mod_sites:
            if num_bases <= mod_loc_in_read < len(seq) - num_bases:
                if strand == "-":
                    pos = chrom_start + len(seq) - mod_loc_in_read - 1
                    pos_in_ref = chrom_len - pos - 1 if chrom_len > 0 else -1
                else:
                    pos = chrom_start + mod_loc_in_read
                    pos_in_ref = pos if chrom_len > 0 else -1
                # TODO: add positions in reference

                k_mer = seq[mod_loc_in_read - num_bases: mod_loc_in_read + num_bases + 1]
                k_mer_signal = signal_list[mod_loc_in_read - num_bases: mod_loc_in_read + num_bases + 1]
                signal_lens = [len(x) for x in k_mer_signal]

                # TODO: add features
                signal_means = [np.mean(x) for x in k_mer_signal]
                signal_stds = [np.std(x) for x in k_mer_signal]
                signal_skews = [robust_skewness(x)[0] for x in k_mer_signal]
                signal_kurts = [robust_kurtosis(x)[0] for x in k_mer_signal]
                signal_max = [np.max(x) for x in k_mer_signal]
                signal_min = [np.min(x) for x in k_mer_signal]
                signal_median = [np.median(x) for x in k_mer_signal]

                features = {
                    "chrom": chrom,
                    "strand": strand,
                    "pos": pos,
                    "pos_in_ref": pos_in_ref,
                    "k_mer": k_mer,
                    "signal_lens": signal_lens,
                    "signal_means": signal_means,
                    "signal_stds": signal_stds,
                    "signal_skews": signal_skews,
                    "signal_kurts": signal_kurts,
                    "signal_max": signal_max,
                    "signal_min": signal_min,
                    "signal_median": signal_median,
                    "methyl_label": methyl_label
                }
                features_list.append(features)
    return features_list


def _features_to_str(features):
    """
    Convert features to string.
    todo 转换特征为字符串
    """
    return None


def _extract_batch_features(
        fast5s_queue, features_queue, error_queue, corrected_group, basecall_subgroup, ref,
        motif_seqs, mod_loc, kmers, methyl_label
):
    while True:
        fast5s = fast5s_queue.get()
        if fast5s is None:
            break
        try:
            features_list = _extract_features(
                fast5s, corrected_group, basecall_subgroup, ref,
                motif_seqs, mod_loc, kmers, methyl_label
            )
            features_strs = [_features_to_str(features) for features in features_list]
            features_queue.put(features_strs)
            while features_queue.qsize() > queue_size_border:
                time.sleep(sleep_time)
        except Exception as e:
            error_queue.put(e)
            break


def _write_features(
        features_queue, output_dir, overwrite
):
    """
    todo 将特征写入磁盘
    """
    pass


def extract_features(
        fast5_dir, recursive, corrected_group, basecall_subgroup, ref_path,
        motifs, mod_loc, kmers, methyl_label,
        output_dir, overwrite,
        processes, batch_size
):
    """
    Extract features from data files.
    """
    if kmers % 2 == 0:
        raise ValueError("kmers must be odd.")
    ref = DNAReference(ref_path)

    start = time.time()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fast5s_queue, motif_seqs, num_fast5s = _preprocess(
        fast5_dir, recursive, motifs, batch_size
    )
    mod_loc = [int(x) for x in mod_loc.split(",")]
    features_queue = Queue()
    error_queue = Queue()

    # Need a process to write the features to disk.
    if processes > 1:
        processes -= 1

    # Start the processes to extract features.
    fast5s_queue.put(None)
    features_processes = []
    for i in range(processes):
        p = mp.Process(
            target=_extract_batch_features,
            args=(
                fast5s_queue, features_queue, error_queue, corrected_group, basecall_subgroup, ref,
                motif_seqs, mod_loc, kmers, methyl_label
            )
        )
        p.daemon = True
        p.start()
        features_processes.append(p)

    # Write the features to disk.
    p_write = mp.Process(
        target=_write_features,
        args=(
            features_queue, output_dir, overwrite
        )
    )
    p_write.daemon = True
    p_write.start()

    error_sum = 0
    while True:
        running = any([p.is_alive() for p in features_processes])
        while not error_queue.empty():
            error_sum += 1
            print(error_queue.get())
        if not running:
            break

    for p in features_processes:
        p.join()
    features_queue.put(None)
    p_write.join()

    print("Extracted features from {} data files in {} seconds.".format(num_fast5s, time.time() - start))
    print("Encountered {} errors.".format(error_sum))


def main():
    extraction_parser = argparse.ArgumentParser(
        "Extract features from data files corrected by Tombo."
    )
    ep_input = extraction_parser.add_argument_group("Input")
    ep_input.add_argument(
        "--fast5_dir", "-i", type=str, required=True, action="store",
        help="Path to data files corrected by Tombo."
    )
    ep_input.add_argument(
        "--recursive", "-r", action="store_true", required=False, default=True,
        help="Recursively search for data files in the input directory."
    )
    ep_input.add_argument(
        "--corrected_group", "-c", type=str, required=False, default="RawGenomeCorrected_000",
        help="The name of the corrected group in the data files."
    )
    ep_input.add_argument(
        "--basecall_subgroup", "-b", type=str, required=False, default="BaseCalled_template",
        help="The name of the basecall subgroup in the data files."
    )
    ep_input.add_argument(
        "--ref_path", "-ref", type=str, required=True, action="store",
        help="Path to ref_path fasta file."
    )

    ep_extraction = extraction_parser.add_argument_group("Extraction")
    ep_extraction.add_argument(
        "--motifs", "-m", type=str, required=False, action="store", default="CG",
        help="The motifs to extract features for. "
             "Multiple motifs should be separated by commas."
             "IUPAC codes are supported."
    )
    ep_extraction.add_argument(
        "--mod_loc_in_motif", "-mlm", type=int, required=False, action="store", default=0,
        help="The location of the modified base in the motifs. "
    )
    ep_extraction.add_argument(
        "--kmers", "-k", type=int, required=False, action="store", default=7,
        help="The number of kmers to extract features for."
    )
    ep_extraction.add_argument(
        "--methyl_label", "-ml", type=int, required=False, action="store", default=1, choices=[0, 1],
        help="The label for the methylated state. This is for training purposes only."
    )

    ep_output = extraction_parser.add_argument_group("Output")
    ep_output.add_argument(
        "--output_dir", "-o", type=str, required=True, action="store",
        help="Path to output directory."
    )
    ep_output.add_argument(
        "--overwrite", "-w", action="store_true", required=False, default=False,
        help="Overwrite existing output files."
    )

    extraction_parser.add_argument(
        "--processes", "-p", type=int, required=False, action="store", default=1,
        help="The number of processes to use for feature extraction."
    )
    extraction_parser.add_argument(
        "--batch_size", "-bs", type=int, required=False, action="store", default=100,
        help="The number of data files to process in each batch."
    )

    args = extraction_parser.parse_args()
    fast5_dir = args.fast5_dir
    recursive = args.recursive
    corrected_group = args.corrected_group
    basecall_subgroup = args.basecall_subgroup
    ref_path = args.reference

    motifs = args.motifs
    mod_loc_in_motifs = args.mod_loc_in_motifs
    kmers = args.kmers
    methyl_label = args.methyl_label

    output_dir = args.output_dir
    overwrite = args.overwrite

    processes = args.processes
    batch_size = args.batch_size

    extract_features(
        fast5_dir, recursive, corrected_group, basecall_subgroup, ref_path,
        motifs, mod_loc_in_motifs, kmers, methyl_label,
        output_dir, overwrite,
        processes, batch_size
    )


if __name__ == "__main__":
    main()
