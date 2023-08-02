import os

import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from torch.utils.data import DataLoader

from nano import dataloader
from nano.models import ModelBiLSTM

model_path = "/home/eason/ont/data/train/model/model_001_kmer7/model.pt"
test_path = "/home/eason/ont/data/train/kmer7/test2.txt"

model = ModelBiLSTM(kmer_len=7)
model.load_state_dict(torch.load(model_path))
print("Model loaded")

model = model.cuda()
model.eval()

dataloader = DataLoader(
    dataloader.SignalFeatureData(test_path),
    batch_size=4096, shuffle=False
)
print("Data loaded")

accuracy = []
df = pd.DataFrame()
print("#" * 20)
print("Start predicting")
for i, data in enumerate(dataloader):
    info, features, labels = data
    features = [f.cuda() for f in features]

    pred = model(features)
    pred = pred.cpu().argmax(dim=1)
    df = pd.concat([df, pd.DataFrame({"info": info, "label": labels, "pred": pred})])
    accuracy.append(metrics.accuracy_score(labels, pred))
    print("Batch: {}, Accuracy: {}".format(i, accuracy[-1]))
print("#" * 20)
print("Accuracy: {}".format(np.mean(accuracy)))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.boxplot(accuracy, ax=axes[0])
axes[0].set_title("Accuracy: {}".format(np.mean(accuracy)))

cm = metrics.confusion_matrix(df["label"], df["pred"], labels=[0, 1])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Unmethylated', 'Methylated'], yticklabels=['Unmethylated', 'Methylated'], ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('True')

plt.savefig(os.path.join(os.path.dirname(test_path), "test2.png"))
df.to_csv(os.path.join(os.path.dirname(test_path), "test2_pred.csv"), index=False)
