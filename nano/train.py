import argparse
import os.path
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


def valid(model, valid_dataloader, criterion):
    model.eval()
    y_true, y_pred = [], []
    loss = 0
    with torch.no_grad():
        for i, data in enumerate(valid_dataloader):
            info, features, labels = data
            if USE_CUDA:
                features = [f.cuda() for f in features]
                labels = labels.cuda()
            pred = model(features)
            loss += criterion(pred, labels).item()
            if USE_CUDA:
                labels = labels.cpu()
                pred = pred.cpu()
            y_true.extend(labels.numpy())
            y_pred.extend(torch.argmax(pred, dim=1).numpy())
    model.train()
    return loss / len(valid_dataloader), metrics.accuracy_score(y_true, y_pred)


def train(args):
    torch.manual_seed(args.seed)
    if USE_CUDA:
        torch.cuda.manual_seed(args.seed)
        logger.info("Using GPU")
    else:
        logger.info("Using CPU")
    logger.info("Loading data")
    train_dataloader = DataLoader(
        SignalFeatureData(args.train_file),
        batch_size=args.batch_size, shuffle=True,
    )
    valid_dataloader = DataLoader(
        SignalFeatureData(args.valid_file),
        batch_size=args.batch_size, shuffle=True,
    )

    logger.info("Loading model")
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model = ModelBiLSTM(
        model_type=args.model_type,
        kmer_len=args.kmer_len,
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
        model.load_state_dict(params)

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
        train_loss = []
        for i, data in enumerate(train_dataloader):
            _, features, labels = data
            if USE_CUDA:
                features = [f.cuda() for f in features]
                labels = torch.LongTensor(labels).cuda()

            # forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            train_loss.append(loss.detach().item())

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            if (i + 1) % args.log_interval == 0 or (i + 1) == train_step:
                model.eval()
                valid_loss, valid_accuracy = valid(model, valid_dataloader, criterion)
                if valid_accuracy > best_accuracy:
                    best_accuracy = valid_accuracy
                    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
                    logger.info("Save model at epoch {}".format(epoch + 1))
                logger.info(
                    "Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}, Valid Accuracy: {:.4f}".format(
                        epoch + 1, args.num_epochs, i + 1, train_step,
                        np.mean(train_loss), valid_loss, valid_accuracy
                    ))
                train_loss = []
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
        "--kmer_len", type=int, default=9, required=False,
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
        "--tmp_dir", type=str, default="/tmp", required=False,
        help="tmp dir, default: /tmp"
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
