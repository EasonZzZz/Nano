"""
this file is used to extract features from the dataset.
output: a csv file with the following columns:
    read_id, chrom, pos, strand, kmer,
    signal_mean, signal_std, signal_skew, signal_kurt,
    signal_length, signals, methyl_label
"""
import glob
import os
import argparse
import random
import shutil
import time
import warnings

import h5py
import numpy as np
import pandas as pd
import multiprocessing as mp

from statsmodels.robust import mad
from statsmodels.stats.stattools import robust_kurtosis, robust_skewness

from nano.utils.constant import QUEUE_BORDER_SIZE, SLEEP_TIME, GLOBAL_KEY, READS_GROUP
from nano.utils.process_utils import get_fast5s, get_motif_seqs, get_ref_loc_of_methyl_site, Queue
from nano.utils.ref_helper import DNAReference
from nano.utils import logging

warnings.filterwarnings("ignore")
logger = logging.get_logger(__name__)
random.seed(42)


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
    except KeyError:
        logger.debug("Could not find status under Analyses/{}/{}/"
                     " in data file: {}".format(corrected_group, basecall_subgroup, fast5_file))
        align_status = None

    if align_status != "success":
        logger.debug("Alignment status is not success for file: {}".format(fast5_file))
        return None, None, None

    info = {}
    try:
        raw_signal = list(fast5_data[READS_GROUP].values())[0]
        info["read_id"] = raw_signal.attrs["read_id"].decode("utf-8")
        raw_signal = raw_signal["Signal"][()]
    except Exception:
        fast5_data.close()
        raise RuntimeError("Could not find raw signal under Raw/Reads/Read_[read#]"
                           " in data file: {}".format(fast5_file))

    try:
        channel_info = dict(list(fast5_data[GLOBAL_KEY]["channel_id"].attrs.items()))
        scaling = channel_info["range"] / channel_info["digitisation"]
        offset = channel_info["offset"]
        raw_signal = _rescale_signals(raw_signal, scaling, offset)
    except Exception:
        fast5_data.close()
        raise RuntimeError("Could not find channel info under UniqueGlobalKey/channel_id"
                           " in data file: {}".format(fast5_file))

    try:
        alignment = dict(list(fast5_data["Analyses"][corrected_group][basecall_subgroup]["Alignment"].attrs.items()))
        info["strand"] = alignment["mapped_strand"]
        info["chrom"] = alignment["mapped_chrom"]
        info["chrom_start"] = alignment["mapped_start"]
        info["chrom_end"] = alignment["mapped_end"]
    except Exception:
        fast5_data.close()
        raise RuntimeError("Could not find strand under Analyses/{}/{}/Alignment"
                           " in data file: {}".format(corrected_group, basecall_subgroup, fast5_file))

    try:
        events = fast5_data["Analyses"][corrected_group][basecall_subgroup]["Events"]
    except Exception:
        fast5_data.close()
        raise RuntimeError("Could not find events under Analyses/{}/{}/Events"
                           " in data file: {}".format(corrected_group, basecall_subgroup, fast5_file))

    try:
        events_attrs = dict(list(events.attrs.items()))
        read_start_rel_to_raw = events_attrs["read_start_rel_to_raw"]
        starts = list(map(lambda x: x + read_start_rel_to_raw, events["start"]))
    except Exception:
        fast5_data.close()
        raise RuntimeError("Could not find read_start_rel_to_raw attribute under Analyses/{}/{}/Events"
                           " in data file: {}".format(corrected_group, basecall_subgroup, fast5_file))

    lengths = events["length"].astype(np.int)
    base = [x.decode("utf-8") for x in events["base"]]
    assert len(starts) == len(lengths) == len(base)
    events = pd.DataFrame({"start": starts, "length": lengths, "base": base})
    fast5_data.close()
    return raw_signal, events, info


def normalize_signal(raw_signal):
    """
    Normalize the raw signal.
    """
    # try z-sore
    # return (raw_signal - np.mean(raw_signal)) / np.std(raw_signal)

    # try mad
    return (raw_signal - np.median(raw_signal)) / mad(raw_signal)


def _fill_fast5s_queue(fast5s_queue, fast5s, batch_size):
    """
    Fill the fast5 queue.
    """
    for i in range(0, len(fast5s), batch_size):
        fast5s_queue.put(fast5s[i:i + batch_size])
    return fast5s_queue


def _preprocess(fast5_dir, recursive, motifs, batch_size):
    """
    Preprocess the fast5 files.
    """
    fast5s = get_fast5s(fast5_dir, recursive)
    if len(fast5s) == 0:
        raise RuntimeError("No fast5 files found in directory: {}".format(fast5_dir))
    logger.info("Found {} fast5 files.".format(len(fast5s)))
    fast5s_queue = _fill_fast5s_queue(Queue(), fast5s, batch_size)

    motif_seqs = get_motif_seqs(motifs)

    return fast5s_queue, motif_seqs, len(fast5s)


def _format_signal(signals_list, signal_len=16):
    """
    Turn the signals into a fixed length list.
    """
    signals = []
    for signal in signals_list:
        signal = list(np.around(signal, 6))
        if len(signal) < signal_len:
            pad0_len = signal_len - len(signal)
            pad0_left = pad0_len // 2
            pad0_right = pad0_len - pad0_left
            signal = [0] * pad0_left + signal + [0] * pad0_right
        elif len(signal) > signal_len:
            signal = [signal[i] for i in sorted(random.sample(range(len(signal)), signal_len))]
        signals.append(signal)
    return signals


def _extract_features(
        fast5s, error_queue, failed_align, corrected_group, basecall_subgroup, ref,
        motif_seqs, mod_loc, kmer_len, methyl_label, positions
):
    """
    Extract features from fast5 files.
    """
    num_bases = (kmer_len - 1) // 2

    failed_counter, error_counter = 0, 0
    features = pd.DataFrame()
    for fast5 in fast5s:
        try:
            raw_signal, events, info = get_raw_signal(fast5, corrected_group, basecall_subgroup)
            if raw_signal is None:
                failed_counter += 1
                continue

            raw_signal = normalize_signal(raw_signal)
            seq, signal_list = "", []
            for _, row in events.iterrows():
                seq += row["base"]
                signal_list.append(raw_signal[row["start"]: row["start"] + row["length"]])

            strand, chrom = info["strand"], info["chrom"]
            chrom_start, chrom_end = info["chrom_start"], info["chrom_end"]
            try:
                chrom_len = ref.get_chrom_length(chrom)
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

                    if (positions is not None) and ("\t".join([chrom, str(pos_in_ref)]) not in positions):
                        continue

                    kmers = seq[mod_loc_in_read - num_bases: mod_loc_in_read + num_bases + 1]
                    kmer_signals = signal_list[mod_loc_in_read - num_bases: mod_loc_in_read + num_bases + 1]
                    signal_lens = [len(x) for x in kmer_signals]

                    # adding features
                    signal_means = [np.mean(x) for x in kmer_signals]
                    signal_stds = [np.std(x) for x in kmer_signals]
                    signal_skews = [robust_skewness(x)[0] if len(x) > 1 else 0 for x in kmer_signals]
                    signal_kurts = [robust_kurtosis(x)[0] if len(x) > 1 else 0 for x in kmer_signals]
                    signals = _format_signal(kmer_signals)

                    feature_dict = {
                        "read_id": info["read_id"],
                        "chrom": chrom,
                        "pos": pos,
                        "strand": strand,
                        "kmer": kmers,
                        "signal_mean": signal_means,
                        "signal_std": signal_stds,
                        "signal_skew": signal_skews,
                        "signal_kurt": signal_kurts,
                        "signal_length": signal_lens,
                        "signals": signals,
                        "methyl_label": methyl_label
                    }
                    features = features.append(feature_dict, ignore_index=True)
        except Exception:
            error_counter += 1
            logger.error("Error occurred when processing file: {}".format(fast5))
            continue
    error_queue.put(error_counter)
    failed_align.put(failed_counter)

    return features


def _extract_batch_features(
        fast5s_queue, features_queue, error_queue, failed_align, corrected_group, basecall_subgroup, ref,
        motif_seqs, mod_loc, kmer_len, methyl_label, positions
):
    while True:
        fast5s = fast5s_queue.get()
        if fast5s is None:
            fast5s_queue.put(None)
            break
        features_list = _extract_features(
            fast5s, error_queue, failed_align, corrected_group, basecall_subgroup, ref,
            motif_seqs, mod_loc, kmer_len, methyl_label, positions
        )
        features_queue.put(features_list)
        while features_queue.qsize() > QUEUE_BORDER_SIZE:
            time.sleep(SLEEP_TIME)


def _write_features(
        features_queue, output_dir, output_batch_size
):
    df_buffer = pd.DataFrame()
    counter = 0
    while True:
        features = features_queue.get()
        if features is None:
            break
        if len(features) == 0:
            continue
        df_buffer = pd.concat([df_buffer, features], axis=0)
        output_path_file = os.path.join(output_dir, "features_{}.csv".format(counter))
        if len(df_buffer) >= output_batch_size:
            df_write = df_buffer.iloc[:output_batch_size]
            df_buffer = df_buffer.iloc[output_batch_size:]
            df_write.to_csv(output_path_file, index=False, header=True)
            logger.info("Wrote features to {}.".format(output_path_file))
            counter += 1
    if len(df_buffer) > 0:
        output_path_file = os.path.join(output_dir, "features_{}.csv".format(counter))
        df_buffer.to_csv(output_path_file, index=False, header=True)
        logger.info("Wrote features to {}.".format(output_path_file))
    logger.info("Finished writing features to {}.".format(output_dir))


def extract_features(
        fast5_dir, recursive, corrected_group, basecall_subgroup, ref_path,
        motifs, mod_loc, kmer_len, methyl_label, positions_file,
        output_dir, overwrite, output_batch_size,
        processes, batch_size
):
    """
    Extract features from fast5 files.
    """
    if kmer_len % 2 == 0:
        raise ValueError("kmer_len must be odd.")
    ref = DNAReference(ref_path)
    if positions_file is not None:
        positions = pd.read_csv(positions_file, sep="\t", names=["chrom", "pos"], header=None)
        positions["pos"] = positions["pos"].astype(str)
        positions = set(positions.apply(lambda x: "\t".join(x), axis=1))
    else:
        positions = None

    if os.path.exists(output_dir):
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise RuntimeError("Output file already exists: {}".format(output_dir))
    os.makedirs(output_dir)
    # logging.init_logger(log_file=os.path.join(output_dir, "extract_features.log"))

    start = time.time()
    fast5s_queue, motif_seqs, num_fast5s = _preprocess(
        fast5_dir, recursive, motifs, batch_size
    )
    features_queue = Queue()
    error_queue = Queue()
    failed_align = Queue()

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
                fast5s_queue, features_queue, error_queue, failed_align, corrected_group, basecall_subgroup, ref,
                motif_seqs, mod_loc, kmer_len, methyl_label, positions
            )
        )
        p.daemon = True
        p.start()
        features_processes.append(p)

    # Write the features to disk.
    p_write = mp.Process(
        target=_write_features,
        args=(
            features_queue, output_dir, output_batch_size
        )
    )
    p_write.daemon = True
    p_write.start()

    error_sum = 0
    while True:
        running = any(p.is_alive() for p in features_processes)
        while not error_queue.empty():
            error_sum += error_queue.get()
        if not running:
            break

    for p in features_processes:
        p.join()
    features_queue.put(None)
    p_write.join()

    failed_counter = 0
    while not failed_align.empty():
        failed_counter += failed_align.get()

    logger.info("Extracted features from {} fast5 files in {:.2f} seconds.".format(num_fast5s, time.time() - start))
    logger.info("Failed to align {} reads.".format(failed_counter))
    logger.info("Encountered {} errors.".format(error_sum))


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
        "--kmer_len", "-k", type=int, required=False, action="store", default=9,
        help="The length of the kmer to extract features for."
    )
    ep_extraction.add_argument(
        "--methyl_label", "-ml", type=int, required=False, action="store", default=1, choices=[0, 1],
        help="The label for the methylated state. This is for training purposes only."
    )
    ep_extraction.add_argument(
        "--positions", "-pos", type=str, required=False, action="store", default=None,
        help="A tab-separated file containing the positions_file interested."
             "The first column is the chromosome, the second column is the position."
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
        "--output_batch_size", "-obs", type=int, required=False, action="store", default=100000,
        help="The number of features to write to disk in each batch."
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
    ref_path = args.reference

    motifs = args.motifs
    mod_loc_in_motif = args.mod_loc_in_motif
    kmer_len = args.kmer_len
    methyl_label = args.methyl_label
    positions_file = args.positions

    output_dir = args.output_dir
    overwrite = args.overwrite
    output_batch_size = args.output_batch_size

    processes = args.processes
    batch_size = args.batch_size

    extract_features(
        fast5_dir, recursive, corrected_group, basecall_subgroup, ref_path,
        motifs, mod_loc_in_motif, kmer_len, methyl_label, positions_file,
        output_dir, overwrite, output_batch_size,
        processes, batch_size
    )


if __name__ == "__main__":
    main()
