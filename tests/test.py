import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch

from nano import extract_features
from nano import dataloader
from nano.models import ModelBiLSTM
from nano.utils.constant import USE_CUDA

data_dir = "../data"
output_dir = "../output/features"


class MyTestCase(unittest.TestCase):
    def test_get_raw_signal(self):
        fast5s = extract_features.get_fast5s(fast5_dir=data_dir, recursive=True)
        raw, events, info = extract_features.get_raw_signal(
            fast5_file=fast5s[0],
            corrected_group="RawGenomeCorrected_000",
            basecall_subgroup="BaseCalled_template",
        )
        self.assertIsNotNone(raw)
        plt.plot(raw)

        self.assertIsNotNone(events)
        self.assertIsNotNone(info)
        print(events)
        print(info)

    def test_dataloader(self):
        dataset = dataloader.SignalFeatureData(data_dir=output_dir)
        self.assertNotEqual(len(dataset), 0)
        print(len(dataset))
        print(dataset[0])
        print(dataset.get_info(0))

    def test_model(self):
        model = ModelBiLSTM()
        if USE_CUDA:
            model.cuda()
        dataset = dataloader.SignalFeatureData(data_dir=output_dir)
        self.assertNotEqual(len(dataset), 0)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
        for i, data in enumerate(train_loader):
            info, features, labels = data
            if USE_CUDA:
                features = [f.cuda() for f in features]
                labels = labels.cuda()
            pred = model(features)
            print(pred[0])
            print(torch.argmax(pred[0]))
            break

    def test(self):
        self.assertEqual(True, True)
        from statsmodels.stats.stattools import robust_kurtosis, robust_skewness
        print(robust_skewness([1])[0] == np.nan)


if __name__ == '__main__':
    unittest.main()
