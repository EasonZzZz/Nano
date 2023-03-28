import glob
import os
import shutil
import sys
import time
import argparse
import uuid

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.multiprocessing.queue import Queue

from nano.extract_features import extract_features
from nano.models import ModelBiLSTM
from nano.utils import logging
from nano.utils.constant import USE_CUDA, QUEUE_BORDER_SIZE, SLEEP_TIME

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

os.environ['MKL_THREADING_LAYER'] = 'GNU'
logger = logging.get_logger(__name__, level=logging.INFO)


def _divide_batches(features, batch_size):
    queue = Queue()
    for i in range(0, len(features), batch_size):
        queue.put(features[i:i + batch_size])
    queue.put(None)
    return queue


def _call_modifications(model, features_batch, batch_size, device):
    for i in np.arange(0, features_batch.shape[0], batch_size):
        features = features_batch[i:i + batch_size]
        info = features[:, :4]
        features = features[:, 4:]
        features = torch.from_numpy(features).float()
        if USE_CUDA:
            features = features.cuda(device)
        with torch.no_grad():
            pred = model(features)

        pred = pred.cpu().numpy()
        yield pred


def _call_modifications_gpu_worker(features_batch_q, pred_q, model_path, args, device=0):
    logger.info("Start calling modifications worker-{}".format(os.getpid()))
    start_time = time.time()
    model = ModelBiLSTM(
        model_type=args.model_type,
        seq_len=args.seq_len,
        signal_len=args.signal_len,
        num_combine_layers=args.num_combine_layers,
        num_pre_layers=args.num_pre_layers,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
        num_classes=args.num_classes,
        vocab_size=args.vocab_size,
        embedding_size=args.embedding_size,
        using_base=args.using_base,
        using_signal_len=args.using_signal_len,
    )
    model.load_state_dict(torch.load(model_path, map_location="cuda:{}".format(device)))
    if USE_CUDA:
        model = model.cuda(device)
    model.eval()

    batch_num_total = 0
    while True:
        if features_batch_q.empty():
            time.sleep(SLEEP_TIME)
            continue
        features_batch = features_batch_q.get()
        if features_batch is None:
            features_batch_q.put(None)
            break
        pred = _call_modifications(model, features_batch, args.batch_size, device)
        pred_q.put(pred)
        while pred_q.size() > QUEUE_BORDER_SIZE:
            time.sleep(SLEEP_TIME)
        batch_num_total += features_batch.shape[0] // args.batch_size + 1
    logger.info("Calling modifications worker-{} processed {} batches in {:.2f} s".format(
        os.getpid(), batch_num_total, time.time() - start_time))


def _write_modifications(pred_q, output_dir, success_file):
    pass


def _call_modifications_gpu(features, model_path, success_file, args):
    if len(features) == 0:
        logger.error("No features extracted")
        return
    features_batch_q = _divide_batches(features, args.f5_batch_size * 1000)
    pred_q = Queue()

    processes = []
    for i in range(args.processes):
        p = mp.Process(target=_call_modifications_gpu_worker, args=(features_batch_q, pred_q, model_path, args))
        p.daemon = True
        p.start()
        processes.append(p)

    w_p = mp.Process(target=_write_modifications, args=(pred_q, args.output, success_file))
    w_p.daemon = True
    w_p.start()

    for p in processes:
        p.join()
    pred_q.put(None)
    w_p.join()

    logger.info("Finish calling modifications")


def _call_modifications_cpu(features, model_path, success_file, args):
    pass


def call_modifications(args):
    logger.info("Start calling modifications")
    start_time = time.time()

    model_path = os.path.abspath(args.model)
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file {} does not exist".format(model_path))
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError("Input file {} does not exist".format(input_path))
    success_file = input_path.rstrip("/") + "." + str(uuid.uuid1()) + ".success"
    if os.path.exists(success_file):
        os.remove(success_file)
    if os.path.exists(args.output):
        if args.overwrite:
            shutil.rmtree(args.output)
        else:
            raise FileExistsError("Output directory {} already exists".format(args.output))
    os.makedirs(args.output)
    logging.init_logger(log_path=os.path.join(args.output, "log.txt"))

    features = pd.DataFrame()
    if args.input_type == 0:
        logger.info("Extracting features from {}".format(input_path))
        extract_features(
            fast5_dir=input_path,
            recursive=args.recursive,
            corrected_group=args.corrected_group,
            basecall_subgroup=args.basecall_subgroup,
            ref_path=args.reference,
            motifs=args.motifs,
            mod_loc=args.mod_loc_in_motif,
            kmer_len=args.seq_len,
            methyl_label=-1,
            positions_file=None,
            output_dir=os.path.join(args.output, "features"),
            overwrite=True,
            output_batch_size=args.f5_batch_size * 1000,
            processes=args.processes,
            batch_size=args.f5_batch_size,
        )
        for file in glob.glob(os.path.join(args.output, "features", "features_*.csv"), recursive=True):
            features = pd.concat([features, pd.read_csv(file)])
    else:
        logger.info("Loading features from {}".format(input_path))
        for file in glob.glob(os.path.join(input_path, "features_*.csv"), recursive=True):
            features = pd.concat([features, pd.read_csv(file)])
            logger.info("Found features file: {}".format(file))
    if len(features) == 0:
        logger.error("No features found, please check the input file")
        sys.exit(1)
    if "methyl_label" in features.columns:
        features = features.drop(columns=["methyl_label"])
    print(features.head(10))
    if USE_CUDA:
        _call_modifications_gpu(features, model_path, success_file, args)
    else:
        _call_modifications_cpu(features, model_path, success_file, args)

    logger.info("Finish calling modifications, time used: {:.2f} seconds".format(time.time() - start_time))


def main():
    parser = argparse.ArgumentParser(description='Call modifications')

    input_group = parser.add_argument_group('Input parameters')
    input_group.add_argument(
        '--input', type=str, required=True,
        help="The input file, can be feature files or fast5 files"
        "If is fast5 files, the extraction args should be provided"
    )
    input_group.add_argument(
        '--input_type', type=int, required=False, default=0,
        choices=[0, 1],
        help="The input type, 0 for fast5 files, 1 for feature files"
    )
    input_group.add_argument(
        "--f5_batch_size", type=int, default=100,
        help="The batch size of fast5 files"
    )

    model_group = parser.add_argument_group('Model parameters')
    model_group.add_argument(
        '--model', type=str, required=True,
        help="The model file, should be a .ckpt file"
    )
    model_group.add_argument(
        '--model_type', type=str, required=False, default="Both_BiLSTM",
        choices=["Both_BiLSTM", "Seq_BiLSTM", "Signal_BiLSTM"],
        help="The model type, should be one of the following: "
        "Both_BiLSTM, Seq_BiLSTM, Signal_BiLSTM"
    )
    model_group.add_argument(
        '--seq_len', type=int, required=False, default=9,
        help="The length of the kmer"
    )
    model_group.add_argument(
        '--signal_len', type=int, required=False, default=16,
        help="The length of the signal per base in kmer"
    )
    model_group.add_argument(
        '--num_combine_layers', type=int, required=False, default=2,
        help="The number of layers in the combine BiLSTM"
    )
    model_group.add_argument(
        '--num_pre_layers', type=int, required=False, default=1,
        help="The number of layers in the pre BiLSTM"
    )
    model_group.add_argument(
        '--num_classes', type=int, required=False, default=2,
        help="The number of classes"
    )
    model_group.add_argument(
        '--dropout', type=float, required=False, default=0.5,
        help="The dropout rate"
    )
    model_group.add_argument(
        '--hidden_size', type=int, required=False, default=256,
        help="The hidden size of the BiLSTM"
    )
    model_group.add_argument(
        '--vocab_size', type=int, required=False, default=16,
        help="The vocab size of the kmer"
    )
    model_group.add_argument(
        '--embedding_size', type=int, required=False, default=4,
        help="The embedding size of the kmer"
    )
    model_group.add_argument(
        "--using_base", type=bool, default=True, required=False,
        help="using base or not, default: True"
    )
    model_group.add_argument(
        "--using_signal_len", type=bool, default=True, required=False,
        help="using signal length or not, default: True"
    )
    model_group.add_argument(
        '--batch_size', type=int, required=False, default=512,
        help="The batch size of the model"
    )

    output_group = parser.add_argument_group('Output parameters')
    output_group.add_argument(
        '--output', type=str, required=True,
        help="The output dir to save the predicted results"
    )
    output_group.add_argument(
        '--overwrite', action='store_true', default=True, required=False,
        help="Whether to overwrite the output file"
    )

    extraction_group = parser.add_argument_group('Extraction parameters')
    extraction_group.add_argument(
        '--recursive', action='store_true', default=True, required=False,
        help="Whether to recursively search for fast5 files"
    )
    extraction_group.add_argument(
        '--corrected_group', type=str, default='RawGenomeCorrected_000',
        required=False, help="The corrected group to extract"
    )
    extraction_group.add_argument(
        '--basecall_subgroup', type=str, default='BaseCalled_template',
        required=False, help="The basecall subgroup to extract"
    )
    extraction_group.add_argument(
        "--motifs", "-m", type=str, required=False, action="store", default="CG",
        help="The motifs to extract features for. "
             "Multiple motifs should be separated by commas."
             "IUPAC codes are supported."
    )
    extraction_group.add_argument(
        "--mod_loc_in_motif", "-mlm", type=int, required=False, action="store", default=0,
        help="The location of the modified base in the motifs. "
    )
    extraction_group.add_argument(
        "--positions", "-pos", type=str, required=False, action="store", default=None,
        help="A tab-separated file containing the positions_file interested."
             "The first column is the chromosome, the second column is the position."
    )
    extraction_group.add_argument(
        '--reference', type=str, required=True, action="store",
        help="The reference genome file, should be a .fa file"
    )

    parser.add_argument(
        "--processes", "-p", type=int, required=False, action="store", default=1,
        help="The number of processes to use for feature extraction."
    )

    args = parser.parse_args()
    logger.info("Arguments: {}".format(args))

    call_modifications(args)


if __name__ == '__main__':
    main()
