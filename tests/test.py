import glob
import multiprocessing
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
from nano.utils.ref_helper import DNAReference

data_dir = "../test_data/data"
features_file = "../test_data/output/features.txt"


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
        dataset = dataloader.SignalFeatureData(features_file)
        self.assertNotEqual(len(dataset), 0)
        print(len(dataset))
        print(dataset[0])

    def test_model(self):
        model = ModelBiLSTM()
        if USE_CUDA:
            model.cuda()
        dataset = dataloader.SignalFeatureData(features_file)
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
        with open("../test_data/train.log") as f:
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

        train = pd.read_csv("../test_data/data/features_0.csv")
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

        cnt = train['methyl_label'].value_counts().sort_values()
        print(cnt[0])

    def test_plot(self):
        self.assertEqual(True, True)
        labels = pd.read_csv("../test_data/pred/test2_labels.csv")
        accuracy = np.load("../test_data/pred/test2_accuracy.npy")

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].boxplot(accuracy)
        axes[0].set_title("Accuracy: {}".format(np.mean(accuracy)))

        cm = metrics.confusion_matrix(labels['true'].to_numpy(), labels['pred'].to_numpy(), labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1],
                    xticklabels=['Unmethylated', 'Methylated'], yticklabels=['Unmethylated', 'Methylated'])
        axes[1].set_title("Confusion Matrix")
        axes[1].legend()

        plt.show()

    def test_positions(self):
        self.assertEqual(True, True)
        df = pd.DataFrame(columns=['chrom', 'pos'])
        names = ['chrom', 'pos', 'end', 'name', '?', 'strand', 'thickStart',
                 'thickEnd', 'itemRgb', 'coverage', 'score']
        mod = pd.read_csv("../test_data/modified_bases.5mC.bed", sep='\t', names=names)
        mod = mod[['chrom', 'pos']]
        mod['pos'] = mod['pos'].astype(str)
        with open("../test_data/25张纸.txt") as f:
            for i in range(25):
                barcode = f.readline().strip()
                pos = [("960-%d" % (i + 1), p) for p in f.readline().strip().split(' ')]
                df = pd.concat([df, pd.DataFrame(pos, columns=['chrom', 'pos'])], axis=0)
        df.to_csv("../test_data/25_barcode_methyl.csv", index=False)
        mod = pd.concat([mod, df, df], axis=0).drop_duplicates(keep=False)
        mod.to_csv("../test_data/25_barcode_unmethyl.csv", index=False)

    def test_wrong_kmer(self):
        self.assertEqual(True, True)
        # df = pd.read_csv("../test_data/pred/test2_pred.csv", index_col=0)
        # df.reset_index(inplace=True)
        # df[['read_id', 'chrom', 'pos', 'strand']] = df['info'].str.split('\t', expand=True)
        # df.drop(['info'], axis=1, inplace=True)
        # ref = DNAReference("../test_data/1013.fa")
        # df['kmer'] = df.apply(lambda x: ref.get_chrom_seq(x['chrom'])[int(x['pos']) - 4:int(x['pos']) + 6], axis=1)
        # df.to_csv("../test_data/pred/test2_pred.csv", index=False)
        df = pd.read_csv("../test_data/pred/test1_pred.csv")
        df = df[df['pred'] != df['label']]

        # fig, ax = plt.subplots(figsize=(10, 6))
        # sns.countplot(x='kmer', hue='label', data=df, order=df['kmer'].value_counts().index, ax=ax)
        # plt.xticks(rotation=90)
        # fig.tight_layout()
        # plt.show()
        #
        # fig, ax = plt.subplots(figsize=(10, 6))
        # sns.countplot(x='pos', hue='label', data=df, order=df['pos'].value_counts().index, ax=ax)
        # plt.xticks(rotation=90)
        # fig.tight_layout()
        # plt.show()

        kmer_count = pd.DataFrame(df['kmer'].value_counts()).reset_index()
        kmer_count.columns = ['kmer', 'count']
        auc = np.load("../test_data/pred/mer_type_10_AUC.npy", allow_pickle=True)
        auc = {i[0]: i[1] for i in auc}
        kmer_count['auc'] = kmer_count['kmer'].map(auc).astype(float)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='kmer', y='auc', data=kmer_count, ax=ax, color='#1f77b4')
        ax.set_ylim(0, 1)
        plt.xticks(rotation=90)
        fig.tight_layout()
        plt.show()


if __name__ == '__main__':
    unittest.main()
