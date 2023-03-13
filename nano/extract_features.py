"""
this file is used to extract features from the dataset.
"""

import os
import sys
import argparse
import h5py
import vbz_h5py_plugin
import numpy as np
import pandas as pd
import multiprocessing as mp


reads_group = "Raw/Reads"


def get_raw_signal(fast5_file, corrected_group, basecall_subgroup):
    """
    Get the raw signal from a fast5 file.
    """
    try:
        fast5_data = h5py.File(fast5_file, "r")
    except IOError:
        raise IOError("Could not open fast5 file: {}".format(fast5_file))

    try:
        raw_signal = list(fast5_data[reads_group].values())[0]
        raw_signal = raw_signal["Signal"][()]
    except Exception:
        raise RuntimeError("Could not find raw signal under Raw/Reads/Read_[read#]"
                           " in fast5 file: {}".format(fast5_file))

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
    return raw_signal, events


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
        "--nproc", "-p", type=int, required=False, default=1,
        help="Number of processes to use for feature extraction."
    )

    ep_extraction = extraction_parser.add_argument_group("Extraction")
    ep_extraction.add_argument(
        "--motif", "-m", type=str, required=False, action="store", default="CG",
        help="The motif to extract features for."
    )
    ep_extraction.add_argument(
        "--kmers", "-k", type=int, required=False, action="store", default=3,
        help="The number of kmers to extract features for."
    )


if __name__ == "__main__":
    main()
