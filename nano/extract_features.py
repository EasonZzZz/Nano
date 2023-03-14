"""
this file is used to extract features from the dataset.
"""

import os
import sys
import argparse
import time

import h5py
import vbz_h5py_plugin
import numpy as np
import pandas as pd
import multiprocessing as mp

from nano.utils.process_utils import Queue, get_fast5s, get_motif_seqs

reads_group = "Raw/Reads"
global_key = "UniqueGlobalKey"
queue_size_border = 1000
sleep_time = 1


def _rescale_signals(raw_signal, scaling, offset):
    """
    Rescale the raw signal.
    """
    return np.array(scaling * (raw_signal + offset), dtype=np.float)


def get_raw_signal(fast5_file, corrected_group, basecall_subgroup):
    """
    Get the raw signal from a fast5 file.
    """
    try:
        fast5_data = h5py.File(fast5_file, "r")
    except IOError:
        raise IOError("Could not open fast5 file: {}".format(fast5_file))

    try:
        align_status = fast5_data["Analyses"][corrected_group][basecall_subgroup].attrs["status"]
    except Exception:
        raise RuntimeError("Could not find status under Analyses/{}/{}/"
                           " in fast5 file: {}".format(corrected_group, basecall_subgroup, fast5_file))
    if align_status != "success":
        print("Alignment failed for fast5 file: {}".format(fast5_file), file=sys.stderr)
        return None, None, None

    info = {}
    try:
        raw_signal = list(fast5_data[reads_group].values())[0]
        info["read_id"] = raw_signal.attrs["read_id"].decode("utf-8")
        raw_signal = raw_signal["Signal"][()]
    except Exception:
        raise RuntimeError("Could not find raw signal under Raw/Reads/Read_[read#]"
                           " in fast5 file: {}".format(fast5_file))

    try:
        channel_info = dict(list(fast5_data[global_key]["channel_id"].attrs.items()))
        scaling = channel_info["range"] / channel_info["digitisation"]
        offset = channel_info["offset"]
        raw_signal = _rescale_signals(raw_signal, scaling, offset)
    except Exception:
        raise RuntimeError("Could not find channel info under UniqueGlobalKey/channel_id"
                           " in fast5 file: {}".format(fast5_file))

    try:
        alignment = dict(list(fast5_data["Analyses"][corrected_group][basecall_subgroup]["Alignment"].attrs.items()))
        info["strand"] = alignment["mapped_strand"]
        info["chrom"] = alignment["mapped_chrom"]
        info["chrom_start"] = alignment["mapped_start"]
        info["chrom_end"] = alignment["mapped_end"]
    except Exception:
        raise RuntimeError("Could not find strand under Analyses/{}/{}/Alignment"
                           " in fast5 file: {}".format(corrected_group, basecall_subgroup, fast5_file))

    try:
        events = fast5_data["Analyses"][corrected_group][basecall_subgroup]["Events"]
    except Exception:
        raise RuntimeError("Could not find events under Analyses/{}/{}/Events"
                           " in fast5 file: {}".format(corrected_group, basecall_subgroup, fast5_file))

    try:
        events_attrs = dict(list(events.attrs.items()))
        read_start_rel_to_raw = events_attrs["read_start_rel_to_raw"]
        starts = list(map(lambda x: x + read_start_rel_to_raw, events["start"]))
    except Exception:
        raise RuntimeError("Could not find read_start_rel_to_raw attribute under Analyses/{}/{}/Events"
                           " in fast5 file: {}".format(corrected_group, basecall_subgroup, fast5_file))

    lengths = events["length"].astype(np.int)
    base = [x.decode("utf-8") for x in events["base"]]
    assert len(starts) == len(lengths) == len(base)
    events = pd.DataFrame({"start": starts, "length": lengths, "base": base})
    return raw_signal, events, info


def normalize_signal(raw_signal):
    """
    Normalize the raw signal.
    """
    return (raw_signal - np.mean(raw_signal)) / np.std(raw_signal)


def _fill_fast5s_queue(fast5s_queue, fast5s, batch_size):
    """
    Fill the fast5s queue.
    """
    for i in np.arrange(0, len(fast5s), batch_size):
        fast5s_queue.put(fast5s[i: i + batch_size])
    return fast5s_queue


def _preprocess(fast5_dir, recursive, motifs, batch_size):
    """
    Preprocess the fast5 files.
    """
    fast5s = get_fast5s(fast5_dir, recursive)
    if len(fast5s) == 0:
        raise RuntimeError("No fast5 files found in directory: {}".format(fast5_dir))
    print("Found {} fast5 files.".format(len(fast5s)))
    fast5s_queue = _fill_fast5s_queue(Queue(), fast5s, batch_size)

    motif_seqs = get_motif_seqs(motifs)

    return fast5s_queue, motif_seqs, len(fast5s)


def _extract_features(
        fast5s, corrected_group, basecall_subgroup, reference, motif_seqs, kmers, methyl_label
):
    """
    Extract features from fast5 files.
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

    return 1


def _features_to_str(features):
    """
    Convert features to string.
    todo 转换特征为字符串
    """
    return None


def _extract_batch_features(
        fast5s_queue, features_queue, error_queue,
        corrected_group, basecall_subgroup, reference, motif_seqs, kmers, methyl_label
):
    while True:
        fast5s = fast5s_queue.get()
        if fast5s is None:
            break
        try:
            features_list = _extract_features(
                fast5s, corrected_group, basecall_subgroup, reference, motif_seqs, kmers, methyl_label
            )
            features_strs = [_features_to_str(features) for features in features_list]
            features_queue.put(features_strs)
            while features_queue.qsize() > queue_size_border:
                time.sleep(sleep_time)
        except Exception as e:
            error_queue.put(e)
            break


def _write_features(
        features_queue, output_dir, overwrite, output_batch_size
):
    """
    todo 将特征写入磁盘
    """
    pass


def extract_features(
        fast5_dir, recursive, corrected_group, basecall_subgroup, reference,
        motifs, kmers, methyl_label,
        output_dir, overwrite, output_batch_size,
        processes, batch_size
):
    """
    Extract features from fast5 files.
    """
    if kmers % 2 == 0:
        raise ValueError("kmers must be odd.")

    start = time.time()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fast5s_queue, motif_seqs, num_fast5s = _preprocess(
        fast5_dir, recursive, motifs, batch_size
    )
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
                fast5s_queue, features_queue, error_queue,
                corrected_group, basecall_subgroup, reference, motif_seqs, kmers, methyl_label
            )
        )
        p.daemon = True
        p.start()
        features_processes.append(p)

    # Write the features to disk.
    p_write = mp.Process(
        target=_write_features,
        args=(
            features_queue, output_dir, overwrite, output_batch_size
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

    print("Extracted features from {} fast5 files in {} seconds.".format(num_fast5s, time.time() - start))
    print("Encountered {} errors.".format(error_sum))


def main():
    extraction_parser = argparse.ArgumentParser(
        "Extract features from fast5 files corrected by Tombo."
    )
    ep_input = extraction_parser.add_argument_group("Input")
    ep_input.add_argument(
        "--fast5_dir", "-i", type=str, required=True, action="store",
        help="Path to fast5 files corrected by Tombo."
    )
    ep_input.add_argument(
        "--recursive", "-r", action="store_true", required=False, default=True,
        help="Recursively search for fast5 files in the input directory."
    )
    ep_input.add_argument(
        "--corrected_group", "-c", type=str, required=False, default="RawGenomeCorrected_000",
        help="The name of the corrected group in the fast5 files."
    )
    ep_input.add_argument(
        "--basecall_subgroup", "-b", type=str, required=False, default="BaseCalled_template",
        help="The name of the basecall subgroup in the fast5 files."
    )
    ep_input.add_argument(
        "--reference", "-ref", type=str, required=True, action="store",
        help="Path to reference fasta file."
    )

    ep_extraction = extraction_parser.add_argument_group("Extraction")
    ep_extraction.add_argument(
        "--motifs", "-m", type=str, required=False, action="store", default="CG",
        help="The motifs to extract features for. "
             "Multiple motifs should be separated by commas."
             "IUPAC codes are supported."
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
    ep_output.add_argument(
        "--output_batch_size", "-obs", type=int, required=False, action="store", default=100,
        help="The number of fast5 files to process in each batch."
    )

    extraction_parser.add_argument(
        "--processes", "-p", type=int, required=False, action="store", default=1,
        help="The number of processes to use for feature extraction."
    )
    extraction_parser.add_argument(
        "--batch_size", "-bs", type=int, required=False, action="store", default=100,
        help="The number of fast5 files to process in each batch."
    )

    args = extraction_parser.parse_args()
    fast5_dir = args.fast5_dir
    recursive = args.recursive
    corrected_group = args.corrected_group
    basecall_subgroup = args.basecall_subgroup
    reference = args.reference

    motifs = args.motifs
    kmers = args.kmers
    methyl_label = args.methyl_label

    output_dir = args.output_dir
    overwrite = args.overwrite
    output_batch_size = args.output_batch_size

    processes = args.processes
    batch_size = args.batch_size

    extract_features(
        fast5_dir, recursive, corrected_group, basecall_subgroup, reference,
        motifs, kmers, methyl_label,
        output_dir, overwrite, output_batch_size,
        processes, batch_size
    )


if __name__ == "__main__":
    main()
