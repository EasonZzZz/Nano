import glob
import os
import shutil
import time
import argparse
import uuid

import pandas as pd
import torch

import numpy as np
import torch.multiprocessing as mp

from torch.utils.data import DataLoader

from nano.dataloader import SignalFeatureData
from nano.extract_features import extract_features
from nano.models import ModelBiLSTM
from nano.utils import logging
from nano.utils.constant import USE_CUDA, QUEUE_BORDER_SIZE, SLEEP_TIME, BASE2INT
from nano.utils.process_utils import Queue

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass
os.environ['MKL_THREADING_LAYER'] = 'GNU'
logger = logging.get_logger(__name__, level=logging.INFO)


def _preprocess(features_path):
    features = pd.DataFrame()
    for file in glob.glob(os.path.join(features_path, "features_*.csv")):
        features = pd.concat([features, pd.read_csv(file)], axis=0)
    if len(features) == 0:
        logger.error("No features found in {}".format(features_path))
        return
    features['kmer'] = features['kmer'].apply(lambda x: np.array([BASE2INT[base] for base in x]))
    features['signals'] = features['signals'].apply(lambda x: x.replace('[', '').replace(']', '').split(', '))
    features['signals'] = features['signals'].apply(lambda x: np.array(x).astype(float).reshape(-1, 16))
    for col in features.columns:
        if col in ['kmer', 'signals', 'read_id', 'chrom', 'pos', 'strand']:
            continue
        features[col] = features[col].apply(
            lambda x: np.array(x[1:-1].split(',')).astype(float) if isinstance(x, str) else x
        )
    features.drop(['methyl_label'], axis=1, inplace=True)
    return features


def _call_modifications(model, features_batch, batch_size, device):
    predicts = None
    for i in range(0, len(features_batch), batch_size):
        batch = features_batch[i:i + batch_size]
        info, features = batch
        if USE_CUDA:
            features = [f.cuda(device) for f in features]
        with torch.no_grad():
            logit = model(features)
            pred = torch.argmax(logit, dim=1)
        if USE_CUDA:
            logit = logit.cpu()
            pred = pred.cpu()
        info = np.array(info).reshape(-1, 1)
        logit = logit.data.numpy().reshape(-1, model.num_classes)
        pred = pred.numpy().reshape(-1, 1)
        pred = np.concatenate([info, logit, pred], axis=1)
        if predicts is None:
            predicts = pred
        else:
            predicts = np.concatenate([predicts, pred], axis=0)
    return predicts


def _call_modifications_gpu_worker(features_batch_q, predict_q, model_path, args, device=0):
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
    # model.load_state_dict(torch.load(model_path))
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
        predict_q.put(pred)
        while predict_q.qsize() > QUEUE_BORDER_SIZE:
            time.sleep(SLEEP_TIME)
        batch_num_total += len(features_batch) // args.batch_size + 1
    logger.info("Calling modifications worker-{} processed {} batches in {:.2f} s".format(
        os.getpid(), batch_num_total, time.time() - start_time))


def _write_modifications(pred_q, output_dir, success_file):
    pass


def _call_modifications_gpu(features_path, model_path, success_file, args):
    # loading features
    features = _preprocess(features_path)

    # calling modifications
    predict_q = Queue()
    procs = []
    for i in range(args.processes):
        p = mp.Process(target=_call_modifications_gpu_worker,
                       args=(features[i::args.processes], predict_q, model_path, args, i))
        p.daemon = True
        p.start()
        procs.append(p)

    # writing modifications
    p_w = mp.Process(target=_write_modifications, args=(predict_q, args.output, success_file))
    p_w.daemon = True
    p_w.start()

    for p in procs:
        p.join()
    predict_q.put(None)
    p_w.join()


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
    logging.init_logger(log_file=os.path.join(args.output, "log.txt"))

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
            methyl_label=0,
            positions_file=None,
            output_dir=os.path.join(args.output, "features"),
            overwrite=True,
            output_batch_size=args.f5_batch_size * 1000,
            processes=args.processes,
            batch_size=args.f5_batch_size,
        )
        features_path = os.path.join(args.output, "features")
    else:
        logger.info("Loading features from {}".format(input_path))
        features_path = input_path
    if USE_CUDA:
        _call_modifications_gpu(features_path, model_path, success_file, args)
    else:
        _call_modifications_cpu(features_path, model_path, success_file, args)

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
        "--num_combine_layers", type=int, default=2, required=False,
        help="num of lstm layer for combined seq and signal, default: 2"
    )
    model_group.add_argument(
        "--num_pre_layers", type=int, default=2, required=False,
        help="num of lstm layer for signal (same for seq), default: 2"
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
