import numpy as np
import pandas as pd
import torch

from sklearn import metrics
from nano import dataloader
from nano.models import ModelBiLSTM

model = ModelBiLSTM()
para_dict = torch.load("/home/eason/ont/data/0318_seq/model_0001_over/Both_BiLSTM.b512_s16_epoch4.ckpt",
                       map_location=torch.device('cpu'))
para_dict = para_dict['model_state_dict']
model_dict = model.state_dict()
model_dict.update(para_dict)
model.load_state_dict(model_dict)
del model_dict
print("Model loaded")

model = model.cuda()
model.eval()

dataset = dataloader.SignalFeatureData(data_dir="/home/eason/ont/data/0420_seq/barcode_1/960/features/")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False)
print("Data loaded")

accuracy = []
y_true, y_pred = np.array([]), np.array([])
print("Start predicting")
for i, data in enumerate(dataloader):
    info, features, labels = data
    features = [f.cuda() for f in features]

    pred = model(features)
    pred = pred.cpu()
    pred = pred.argmax(dim=1)
    y_pred = np.concatenate([y_pred, pred.numpy()])
    y_true = np.concatenate([y_true, labels.numpy()])
    accuracy.append(metrics.accuracy_score(labels, pred))

print("Accuracy: {}".format(np.mean(accuracy)))

df = pd.DataFrame({"true": y_true, "pred": y_pred})
df.to_csv("/home/eason/ont/data/0420_seq/barcode_1/960/features/labels.csv", index=False)
np.save("/home/eason/ont/data/0420_seq/barcode_1/960/features/accuracy.npy", np.array(accuracy))

# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.rcParams['figure.figsize'] = (6, 8)
# sns.boxplot(accuracy)
# plt.title("Accuracy: {}".format(np.mean(accuracy)))
# plt.show()
#
# cm = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
# plt.rcParams['figure.figsize'] = (8, 8)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
#             xticklabels=['Unmethylated', 'Methylated'], yticklabels=['Unmethylated', 'Methylated'])
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()
