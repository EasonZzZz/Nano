import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from nano.utils.constant import USE_CUDA


def get_lout(lin, strides):
    import math
    lout = lin
    for s in strides:
        lout = math.floor((lout - 1) / s + 1)
    return lout


class ModelBiLSTM(nn.Module):
    def __init__(
        self, seq_len=9, signal_len=16, num_combine_layers=2, num_pre_layers=2, num_classes=2,
        dropout=0.5, hidden_size=256, vocab_size=16, embedding_size=4,
        using_base=True, using_signal_len=True, model_type="Both_BiLSTM", device=0
    ):
        super(ModelBiLSTM, self).__init__()
        self.model_type = model_type
        self.device = device

        self.seq_len = seq_len
        self.signal_len = signal_len
        self.num_combine_layers = num_combine_layers
        self.num_pre_layers = num_pre_layers
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        if self.model_type == "Both_BiLSTM":
            self.hidden_seq = hidden_size // 2
            self.hidden_signal = self.hidden_size - self.hidden_seq
        elif self.model_type == "Seq_BiLSTM":
            self.hidden_seq = hidden_size
            self.hidden_signal = 0
        elif self.model_type == "Signal_BiLSTM":
            self.hidden_seq = 0
            self.hidden_signal = hidden_size
        else:
            raise ValueError("Invalid module type")

        # Seq BiLSTM
        if self.model_type != "Signal_BiLSTM":
            self.embedding_size = embedding_size
            self.embedding = nn.Embedding(vocab_size, embedding_size)
            self.using_base = using_base
            self.using_signal_len = using_signal_len
            self.signal_feature_num = 5 if self.using_signal_len else 4
            if self.using_base:
                self.lstm_seq = nn.LSTM(
                    input_size=embedding_size + self.signal_feature_num,
                    hidden_size=self.hidden_seq,
                    num_layers=self.num_combine_layers,
                    dropout=dropout,
                    bidirectional=True,
                    batch_first=True
                )
            else:
                self.lstm_seq = nn.LSTM(
                    input_size=self.signal_feature_num,
                    hidden_size=self.hidden_seq,
                    num_layers=self.num_combine_layers,
                    dropout=dropout,
                    bidirectional=True,
                    batch_first=True
                )
            self.fc_seq = nn.Linear(self.hidden_seq * 2, self.hidden_seq)
            self.relu_seq = nn.ReLU()

        # Signal BiLSTM
        if self.model_type != "Seq_BiLSTM":
            self.lstm_signal = nn.LSTM(
                input_size=self.signal_len,
                hidden_size=self.hidden_signal,
                num_layers=self.num_pre_layers,
                dropout=dropout,
                bidirectional=True,
                batch_first=True
            )
            self.fc_signal = nn.Linear(self.hidden_signal * 2, self.hidden_signal)
            self.relu_signal = nn.ReLU()

        # Combine BiLSTM
        self.lstm_combine = nn.LSTM(
            input_size=self.hidden_seq + self.hidden_signal,
            hidden_size=self.hidden_size,
            num_layers=self.num_combine_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.hidden_size, self.num_classes)
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def get_model_type(self):
        return self.model_type

    def _init_hidden(self, batch_size, hidden_size, num_layers):
        h0 = Variable(torch.randn(num_layers * 2, batch_size, hidden_size))
        c0 = Variable(torch.randn(num_layers * 2, batch_size, hidden_size))
        if USE_CUDA:
            h0 = h0.cuda(self.device)
            c0 = c0.cuda(self.device)
        return h0, c0

    def forward(self, data):
        kmer, means, stds, skews, kurts, signal_lens, signals = data
        out_seq, out_signal, out = None, None, None
        # Seq BiLSTM
        if self.model_type != "Signal_BiLSTM":
            means = torch.reshape(means, (-1, self.seq_len, 1)).float()
            stds = torch.reshape(stds, (-1, self.seq_len, 1)).float()
            skews = torch.reshape(skews, (-1, self.seq_len, 1)).float()
            kurts = torch.reshape(kurts, (-1, self.seq_len, 1)).float()
            signal_lens = torch.reshape(signal_lens, (-1, self.seq_len, 1)).float()

            # (batch_size, seq_len, feature_num)
            if self.using_signal_len:
                kmer_embed = self.embedding(kmer.long())
                if self.using_signal_len:
                    out_seq = torch.cat((kmer_embed, means, stds, skews, kurts, signal_lens), dim=2)
                else:
                    out_seq = torch.cat((kmer_embed, means, stds, skews, kurts), dim=2)
            else:
                if self.using_sigal_len:
                    out_seq = torch.cat((means, stds, skews, kurts, signal_lens), dim=2)
                else:
                    out_seq = torch.cat((means, stds, skews, kurts), dim=2)
            # (batch_size, seq_len, hidden_size)
            out_seq, _ = self.lstm_seq(out_seq, self._init_hidden(out_seq.size(0), self.hidden_seq,
                                                                  self.num_combine_layers))
            out_seq = self.fc_seq(out_seq)
            out_seq = self.relu_seq(out_seq)

        # Signal BiLSTM
        if self.model_type != "Seq_BiLSTM":
            out_signal = signals.float()
            # (batch_size, signal_len, hidden_size)
            out_signal, _ = self.lstm_signal(out_signal, self._init_hidden(out_signal.size(0), self.hidden_signal,
                                                                           self.num_pre_layers))
            out_signal = self.fc_signal(out_signal)
            out_signal = self.relu_signal(out_signal)

        if self.model_type == "Both_BiLSTM":
            out = torch.cat((out_seq, out_signal), dim=2)
        elif self.model_type == "Seq_BiLSTM":
            out = out_seq
        elif self.model_type == "Signal_BiLSTM":
            out = out_signal

        if out is None:
            raise ValueError("Model type is not correct!")
        out, _ = self.lstm_combine(out, self._init_hidden(out.size(0), self.hidden_size, self.num_combine_layers))
        out_fwd_last = out[:, -1, :self.hidden_size]
        out_bwd_last = out[:, 0, self.hidden_size:]
        out = torch.cat((out_fwd_last, out_bwd_last), dim=1)

        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.softmax(out)

        return out
