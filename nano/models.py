import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from nano.utils.torch_helper import USE_CUDA


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, strides, out_channel=128, init_channel=1, in_planes=4):
        super(ResNet, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(init_channel, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.layer1 = self._make_layer(block, in_planes, num_blocks[0], stride=strides[0])
        self.layer2 = self._make_layer(block, in_planes*2, num_blocks[1], stride=strides[1])
        self.layer3 = self._make_layer(block, in_planes*4, num_blocks[2], stride=strides[2])

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


def get_lout(lin, strides):
    import math
    lout = lin
    for s in strides:
        lout = math.floor((lout - 1) / s + 1)
    return lout


def ResNet3(in_planes=4, strides=None, out_channel=128, init_channel=1):
    if strides is None:
        strides = [1, 2, 2]
    return ResNet(BasicBlock, [2, 2, 2], strides, out_channel, init_channel, in_planes)


class ModelBiLSTM(nn.Module):
    def __init__(
        self, seq_len=13, signal_len=16, num_layers1=3, num_layers2=1, num_classes=2,
        dropout=0.5, hidden_size=256, vocab_size=16, embedding_size=4,
        using_base=True, using_sigal_len=True, module="Both_BiLSTM", device=0
    ):
        super(ModelBiLSTM, self).__init__()
        self.model_type = "BiLSTM"
        self.module = module
        self.device = device

        self.seq_len = seq_len
        self.signal_len = signal_len
        self.num_layers1 = num_layers1
        self.num_layers2 = num_layers2
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        if self.module == "Both_BiLSTM":
            self.nhid_seq = hidden_size // 2
            self.nhid_signal = self.hidden_size - self.nhid_seq
        elif self.module == "Seq_BiLSTM":
            self.nhid_seq = hidden_size
            self.nhid_signal = 0
        elif self.module == "Signal_BiLSTM":
            self.nhid_seq = 0
            self.nhid_signal = hidden_size
        else:
            raise ValueError("Invalid module type")

        # Seq BiLSTM
        if self.module != "Signal_BiLSTM":
            self.embedding = nn.Embedding(vocab_size, embedding_size)
            self.using_base = using_base
            self.using_sigal_len = using_sigal_len
            self.signal_feature_num = 3 if self.is_signal_len else 2
            if self.is_base:
                self.lstm_seq = nn.LSTM(
                    input_size=embedding_size + self.signal_feature_num,
                    hidden_size=self.nhid_seq,
                    num_layers=self.num_layers1,
                    dropout=dropout,
                    bidirectional=True,
                    batch_first=True
                )
            else:
                self.lstm_seq = nn.LSTM(
                    input_size=self.signal_feature_num,
                    hidden_size=self.nhid_seq,
                    num_layers=self.num_layers1,
                    dropout=dropout,
                    bidirectional=True,
                    batch_first=True
                )
            self.fc_seq = nn.Linear(self.nhid_seq * 2, self.nhid_seq)
            self.relu_seq = nn.ReLU()

        # Signal BiLSTM
        if self.module != "Seq_BiLSTM":
            self.lstm_signal = nn.LSTM(
                input_size=self.signal_len,
                hidden_size=self.nhid_signal,
                num_layers=self.num_layers2,
                dropout=dropout,
                bidirectional=True,
                batch_first=True
            )
            self.fc_signal = nn.Linear(self.nhid_signal * 2, self.nhid_signal)
            self.relu_signal = nn.ReLU()

        # Combine BiLSTM
        self.lstm_combine = nn.LSTM(
            input_size=self.nhid_seq + self.nhid_signal,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers1,
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

    def forward(self, kmer, means, stds, signal_lens, signals):
        # Seq BiLSTM
        if self.module != "Signal_BiLSTM":
            base_means = torch.reshape(means, (-1, self.signal_len)).float()
            base_stds = torch.reshape(stds, (-1, self.signal_len)).float()
            base_signal_lens = torch.reshape(signal_lens, (-1, self.signal_len)).float()
            if self.is_signal_len:
                kmer_embed = self.embedding(kmer.long())
                if self.using_sigal_len:
                    out_seq = torch.cat((kmer_embed, base_means, base_stds, base_signal_lens), dim=2)
                else:
                    out_seq = torch.cat((kmer_embed, base_means, base_stds), dim=2)
            else:
                if self.using_sigal_len:
                    out_seq = torch.cat((base_means, base_stds, base_signal_lens), dim=2)
                else:
                    out_seq = torch.cat((base_means, base_stds), dim=2)
            out_seq, _ = self.lstm_seq(out_seq, self._init_hidden(out_seq.size(0), self.nhid_seq, self.num_layers1))
            out_seq = self.fc_seq(out_seq)
            out_seq = self.relu_seq(out_seq)

        # Signal BiLSTM
        if self.module != "Seq_BiLSTM":
            out_signal = signals.float()
            out_signal, _ = self.lstm_signal(out_signal, self._init_hidden(out_signal.size(0), self.nhid_signal, self.num_layers2))
            out_signal = self.fc_signal(out_signal)
            out_signal = self.relu_signal(out_signal)

        if self.module == "Both_BiLSTM":
            out = torch.cat((out_seq, out_signal), dim=2)
        elif self.module == "Seq_BiLSTM":
            out = out_seq
        elif self.module == "Signal_BiLSTM":
            out = out_signal

        out, _ = self.lstm_combine(out, self._init_hidden(out.size(0), self.hidden_size, self.num_layers1))
        out_fwd_last = out[:, -1, :self.hidden_size]
        out_bwd_last = out[:, 0, self.hidden_size:]
        out = torch.cat((out_fwd_last, out_bwd_last), dim=1)

        out = self.dropout1(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = self.relu2(out)

        return out, self.softmax(out)
