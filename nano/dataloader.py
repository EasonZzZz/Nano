import glob
import os

import pandas as pd

from torch.utils.data import Dataset


class SignalFeatureData(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = self.load_data()

    def load_data(self):
        df = pd.DataFrame()
        for file in glob.glob(os.path.join(self.data_dir, "features_*.csv")):
            df = pd.concat([df, pd.read_csv(file)], axis=0)
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data.iloc[idx])
        return self.data.iloc[idx]
