import glob
import os

import pandas as pd

from torch.utils.data import Dataset


base2code = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
code2base = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: 'N'}


class SignalFeatureData(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        df = pd.DataFrame()
        for file in glob.glob(os.path.join(self.data_dir, "features_*.csv")):
            df = pd.concat([df, pd.read_csv(file)], axis=0)
        df['kmer'] = df['kmer'].apply(lambda x: [base2code[base] for base in x])
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data.iloc[idx])
        return self.data.iloc[idx]
