import glob
import os

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


base2code = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
code2base = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}


class SignalFeatureData(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.info, self.data, self.label = self.load_data()

    def load_data(self):
        df = pd.DataFrame()
        for file in glob.glob(os.path.join(self.data_dir, "features_*.csv")):
            df = pd.concat([df, pd.read_csv(file)], axis=0)
        if len(df) == 0:
            raise ValueError("No data found in {}".format(self.data_dir))
        info = []
        for i, row in df[['read_id', 'chrom', 'pos', 'strand']].iterrows():
            info.append("\t".join([str(i) for i in row.to_numpy()]))
        info = np.array(info)

        data = df.drop(['read_id', 'chrom', 'pos', 'strand', 'methyl_label'], axis=1)
        data['kmer'] = data['kmer'].apply(lambda x: np.array([base2code[base] for base in x]))
        data['signals'] = data['signals'].apply(lambda x: x.replace('[', '').replace(']', '').split(', '))
        data['signals'] = data['signals'].apply(lambda x: np.array(x).astype(float).reshape(-1, 16))
        for col in data.columns:
            if col == 'kmer' or col == 'signals':
                continue
            data[col] = data[col].apply(
                lambda x: np.array(x[1:-1].split(',')).astype(float) if isinstance(x, str) else x
            )

        label = None
        for col in df.columns:
            if 'label' in col:
                label = df[col].to_numpy(dtype=np.int64)
                break

        return info, data, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        output = [i for i in self.data.iloc[idx].to_numpy()]
        if self.transform:
            return self.transform(output)
        return self.info[idx], output, self.label[idx]
