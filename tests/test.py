import glob
import os
import re
import unittest

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import metrics

from nano import extract_features
from nano import dataloader
from nano.models import ModelBiLSTM
from nano.utils.constant import USE_CUDA

data_dir = "../test_data/data"
output_dir = "../test_data/output/features"


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
        train_loss = []
        valid_loss = []
        valid_acc = []
        with open("../train.log") as f:
            for line in f:
                if "Epoch" in line:
                    line = line.split(", ")
                    for l in line:
                        if 'Train Loss' in l:
                            train_loss.append(float(re.findall(r"[-+]?\d*\.\d+|\d+", l)[0]))
                        elif 'Valid Loss' in l:
                            valid_loss.append(float(re.findall(r"[-+]?\d*\.\d+|\d+", l)[0]))
                        elif 'Valid Accuracy' in l:
                            valid_acc.append(float(re.findall(r"[-+]?\d*\.\d+|\d+", l)[0]))
        plt.plot(train_loss, label="train")
        plt.plot(valid_loss, label="valid")
        plt.title("loss")
        plt.legend()
        plt.show()

        plt.plot(valid_acc)
        plt.title("valid accuracy")
        plt.show()

    def test_sample(self):
        self.assertEqual(True, True)

        train = pd.read_csv("../test_data/output/features_0.csv")
        methyl = train[train['methyl_label'] == 1]
        unmethyl = train[train['methyl_label'] == 0]

        cnt = train['kmer'].value_counts().sort_values()
        sns.barplot(y=cnt.values, x=cnt.index)
        plt.xticks(rotation=90, fontsize=5)
        plt.show()

        _methyl = methyl.copy()
        for i in range(len(unmethyl) // len(methyl) - 1):
            methyl = pd.concat([methyl, _methyl], axis=0)

        # unmethyl = unmethyl.sample(n=len(methyl), random_state=42)
        train = pd.concat([methyl, unmethyl], axis=0)
        print(train['methyl_label'].value_counts())

        # base2code = {'A': '0', 'C': '1', 'G': '2', 'T': '3', 'N': '4'}
        # code2base = {v: k for k, v in base2code.items()}
        # x_train = train.drop(["read_id", "chrom", "pos", "strand", 'methyl_label'], axis=1)
        # x_train['kmer'] = x_train['kmer'].apply(lambda x: ''.join([base2code[base] for base in x]))
        # y_train = train['methyl_label']
        # print(y_train.value_counts())
        #
        # rus = RandomUnderSampler(random_state=42)
        # x_train, y_train = rus.fit_resample(x_train, y_train)
        # print(y_train.value_counts())
        #
        # train = pd.concat([x_train, y_train], axis=1)
        # train['kmer'] = train['kmer'].apply(lambda x: ''.join([code2base[base] for base in x]))
        # train['read_id'], train['chrom'], train['pos'], train['strand'] = np.null, np.null, np.null, np.null

    def test_plot(self):
        self.assertEqual(True, True)
        labels = pd.read_csv("../test_data/0420/2/labels.csv")
        accuracy = np.load("../test_data/0420/2/accuracy.npy")

        plt.rcParams['figure.figsize'] = (6, 8)
        sns.boxplot(accuracy)
        plt.title("Accuracy: {}".format(np.mean(accuracy)))
        plt.show()

        cm = metrics.confusion_matrix(labels['true'].to_numpy(), labels['pred'].to_numpy(), labels=[0, 1])
        plt.rcParams['figure.figsize'] = (8, 8)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Unmethylated', 'Methylated'], yticklabels=['Unmethylated', 'Methylated'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()


if __name__ == '__main__':
    unittest.main()
