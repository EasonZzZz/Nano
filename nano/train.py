import argparse
import os.path
import re
import time

import numpy as np
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from sklearn import metrics

from nano.models import ModelBiLSTM
from nano.utils import logging
from nano.utils.constant import USE_CUDA
from nano.dataloader import SignalFeatureData


logger = logging.get_logger(__name__)


def train(args):
    torch.manual_seed(args.seed)
    if USE_CUDA:
        torch.cuda.manual_seed(args.seed)
        logger.info("Using GPU")
    else:
        logger.info("Using CPU")
    logger.info("Loading data")
    if os.path.isfile(args.train_file):
        train_dataset = SignalFeatureData(data_file=args.train_file)
    else:
        train_dataset = SignalFeatureData(data_dir=args.train_file)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    if os.path.isfile(args.valid_file):
        valid_dataset = SignalFeatureData(data_file=args.valid_file)
    else:
        valid_dataset = SignalFeatureData(data_dir=args.valid_file)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False
    )

    logger.info("Loading model")
    model_dir = args.model_dir
    if model_dir != "/":
        model_dir = os.path.abspath(model_dir).rstrip("/")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            model_regex = re.compile(r"" + args.model_type + "\.b\d+_s\d+_epoch\d+\.ckpt*")
            for file in os.listdir(model_dir):
                if model_regex.match(file):
                    os.remove(model_dir + "/" + str(file))
        model_dir += "/"
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
    if USE_CUDA:
        model = model.cuda()
    if args.pretrained_model:
        logger.info("Loading pretrained model: {}".format(args.pretrained_model))
        params = torch.load(args.pretrained_model) if USE_CUDA else torch.load(
            args.pretrained_model, map_location="cpu"
        )
        model.load_state_dict(params["model_state_dict"])

    logger.info("Training model")
    weight = torch.FloatTensor([1, args.pos_weight])
    if USE_CUDA:
        weight = weight.cuda()
    criterion = nn.CrossEntropyLoss(weight=weight)
    logger.info("Using optimizer: {}".format(args.optimizer))
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    else:
        logger.error("Invalid optimizer: {}".format(args.optimizer))
        raise ValueError("Invalid optimizer: {}".format(args.optimizer))
    scheduler = StepLR(optimizer, step_size=2, gamma=.1)

    logger.info("Start training")
    train_step = len(train_dataloader)
    logger.info("Train step: {}".format(train_step))
    best_accuracy, best_epoch = 0, 0
    model.train()
    for epoch in range(args.num_epochs):
        train_losses = []
        for i, data in enumerate(train_dataloader):
            _, features, labels = data
            if USE_CUDA:
                features = [f.cuda() for f in features]
                labels = torch.LongTensor(labels).cuda()

            # forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            train_losses.append(loss.detach().item())

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if (i + 1) % args.log_interval == 0 or (i + 1) == train_step:
                model.eval()
                with torch.no_grad():
                    valid_losses, valid_labels, valid_predicted = [], [], []
                    for vi, vfeatures in enumerate(valid_dataloader):
                        _, vfeatures, vlabels = vfeatures
                        if USE_CUDA:
                            vfeatures = [f.cuda() for f in vfeatures]
                            vlabels = torch.LongTensor(vlabels).cuda()

                        voutputs = model(vfeatures)
                        vloss = criterion(voutputs, vlabels)
                        valid_losses.append(vloss.detach().item())
                        if USE_CUDA:
                            vlabels = vlabels.cpu()
                            voutputs = voutputs.cpu()
                        valid_labels.extend(vlabels.numpy())
                        valid_predicted.extend(voutputs.argmax(dim=1).numpy())
                    valid_accuracy = metrics.accuracy_score(valid_labels, valid_predicted)
                    valid_precision = metrics.precision_score(valid_labels, valid_predicted)
                    valid_recall = metrics.recall_score(valid_labels, valid_predicted)
                    if valid_accuracy > best_accuracy:
                        best_accuracy = valid_accuracy
                        best_epoch = epoch
                        torch.save(model.state_dict(), model_dir + args.model_type + ".b{}_s{}_epoch{}.ckpt".format(
                            args.batch_size, args.seq_len, epoch
                        ))
                        logger.info("Saved a new best model at epoch {} with accuracy {:.4f}"
                                    .format(epoch, valid_accuracy))
                    logger.info(
                        "Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid Accuracy: {:.4f},"
                        " Valid Precision: {:.4f}, Valid Recall: {:.4f}".format(
                            epoch + 1,
                            args.num_epochs,
                            i + 1,
                            train_step,
                            np.mean(train_losses),
                            np.mean(valid_losses),
                            valid_accuracy,
                            valid_precision,
                            valid_recall,
                        )
                    )
                    train_losses = []
                model.train()
        scheduler.step()
    logger.info("Training finished")


def main():
    parser = argparse.ArgumentParser(description='train')

    # required arguments
    required_group = parser.add_argument_group("required arguments")
    required_group.add_argument("--train_file", type=str, required=True, help="train file")
    required_group.add_argument("--valid_file", type=str, required=True, help="valid file")
    required_group.add_argument("--model_dir", type=str, required=True, help="model dir")

    # model parameters
    model_group = parser.add_argument_group("model parameters")
    model_group.add_argument(
        "--model_type", type=str, default="Both_BiLSTM", required=False,
        choices=["Both_BiLSTM", "Seq_BiLSTM", "Signal_BiLSTM"],
        help="type of model to use, default: Both_BiLSTM, choices: Both_BiLSTM, Seq_BiLSTM, Signal_BiLSTM")
    model_group.add_argument(
        "--seq_len", type=int, default=9, required=False,
        help="len of kmers, default: 9"
    )
    model_group.add_argument(
        "--signal_len", type=int, default=16, required=False,
        help="len of signal per base, default: 16"
    )

    # BiLSTM parameters
    bilstm_group = parser.add_argument_group("BiLSTM parameters")
    bilstm_group.add_argument(
        "--num_combine_layers", type=int, default=2, required=False,
        help="num of lstm layer for combined seq and signal, default: 2"
    )
    bilstm_group.add_argument(
        "--num_pre_layers", type=int, default=2, required=False,
        help="num of lstm layer for signal (same for seq), default: 2"
    )
    bilstm_group.add_argument(
        "--num_classes", type=int, default=2, required=False,
        help="num of classes, default: 2"
    )
    bilstm_group.add_argument(
        "--dropout", type=float, default=0.5, required=False,
        help="dropout rate, default: 0.5"
    )
    bilstm_group.add_argument(
        "--hidden_size", type=int, default=256, required=False,
        help="hidden size of lstm, default: 256"
    )
    bilstm_group.add_argument(
        "--vocab_size", type=int, default=16, required=False,
        help="vocab size of kmers, default: 16"
    )
    bilstm_group.add_argument(
        "--embedding_size", type=int, default=4, required=False,
        help="embedding size of kmers, default: 4"
    )
    bilstm_group.add_argument(
        "--using_base", type=bool, default=True, required=False,
        help="using base or not, default: True"
    )
    bilstm_group.add_argument(
        "--using_signal_len", type=bool, default=True, required=False,
        help="using signal length or not, default: True"
    )

    # training parameters
    training_group = parser.add_argument_group("training parameters")
    training_group.add_argument(
        "--optimizer", type=str, default="Adam", required=False,
        choices=["Adam", "SGD"],
        help="optimizer to use, default: Adam, choices: Adam, SGD"
    )
    training_group.add_argument(
        "--lr", type=float, default=0.001, required=False,
        help="learning rate, default: 0.001"
    )
    training_group.add_argument(
        "--batch_size", type=int, default=512, required=False,
        help="batch size, default: 512"
    )
    training_group.add_argument(
        "--num_epochs", type=int, default=10, required=False,
        help="num of epochs, default: 10"
    )
    training_group.add_argument(
        "--pos_weight", type=float, default=1.0, required=False,
        help="pos weight, default: 1.0"
    )
    training_group.add_argument(
        "--seed", type=int, default=42, required=False,
        help="random seed, default: 42"
    )
    training_group.add_argument(
        "--log_interval", type=int, default=100, required=False,
        help="log interval, default: 100"
    )
    training_group.add_argument(
        "--pretrained_model", type=str, default=None, required=False,
        help="pretrained model, default: None"
    )
    training_group.add_argument(
        "--tmp_dir", type=str, default="./tmp", required=False,
        help="tmp dir, default: ./tmp"
    )

    args = parser.parse_args()
    logging.init_logger(log_file=os.path.join(args.model_dir, "train.log"))
    logger.info("training parameters: {}".format(args))
    logger.info("training start...")
    start = time.time()

    train(args)

    end = time.time()
    logger.info("training end, time cost: {}s".format(end - start))


if __name__ == '__main__':
    main()
