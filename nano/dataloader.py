import linecache

import numpy as np

from torch.utils.data import Dataset

from nano.utils.constant import BASE2INT


def str2features(features):
    words = features.split('\t')

    info = "\t".join(words[:4])

    kmers = np.array([BASE2INT[base] for base in words[4]])
    means = np.array([float(i) for i in words[5].split(',')])
    stds = np.array([float(i) for i in words[6].split(',')])
    lens = np.array([float(i) for i in words[7].split(',')])
    signals = np.array([[float(i) for i in j.split(',')] for j in words[8].split(';')])
    features = (kmers, means, stds, lens, signals)

    label = int(words[9])

    return info, features, label


class SignalFeatureData(Dataset):
    def __init__(self, file, transform=None):
        self._file = file
        self._len = 0
        self._transform = transform
        with open(self._file, 'r') as f:
            self._len = len(f.readlines())

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        line = linecache.getline(self._file, idx + 1)
        if line == '':
            return None
        info, features, label = str2features(line)
        if self._transform:
            features = self._transform(features)
        return info, features, label
