import unittest

import matplotlib.pyplot as plt

from nano import extract_features
from nano import dataloader


data_dir = "../data"
output_dir = "../output"


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


if __name__ == '__main__':
    unittest.main()
