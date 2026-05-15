import torch
from torch.utils.data import DataLoader

from network import SleepNet
from dataset import EdfDataset

import sklearn.metrics as skmetrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import argparse

# 命令行传参
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seq_len", type=int, default=64)
parser.add_argument("--network", type=str, default="GRU", help="GRU | LSTM | Attention")
# 要注意这里训练得到的模型要与训练时选用的网络结构相对于，就你用GRU跑的，那自然测试的时候也是用GRU
parser.add_argument("--model_path", type=str, default="./models/model_GRU.pt", help="model path")
parser.add_argument("--data_path", type=str, default="./data/sleepedf/npz", help="model path")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载模型
model = SleepNet(network=args.network, seq_len=args.seq_len)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)

# 加载数据集
data_path = args.data_path
test_dataset = EdfDataset(data_path, seq_len=args.seq_len, is_train=False)
test_dataloader = DataLoader(test_dataset, batch_size=16)

preds = []
trues = []

for batch_idx, (X, y) in enumerate(test_dataloader):
    X, y = X.to(device), y.reshape(-1,).to(device, dtype=torch.long)
    pred = model(X)
    preds.append(pred.argmax(dim=1).cpu())
    trues.append(y.cpu())    

preds = np.hstack(preds)
trues = np.hstack(trues)
acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
f1_score = skmetrics.f1_score(trues,preds,average="macro")
cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0, 1, 2, 3, 4])

print(f"test_acc:{acc*100:.2f}% || test_mf1:{f1_score:.2f}")
print("confusion_matrix:")
print(cm.astype('i'))

# sklearn计算的混淆矩阵的列为预测值，行为真实值

# tp_fp 预测值, 列和
tp_fp = cm.sum(axis=0)  # 每一列求和：预测值统计
# tp_fn 真实值, 行和
tp_fn = cm.sum(axis=1)  # 每一列求和：预测值统计

print("tp_fp", tp_fp)
print("tp_fn", tp_fn)

precision = np.zeros((5,5))
recall = np.zeros((5,5))
for i in range(5):
    recall[i] = cm[i] / tp_fn[i]

for i in range(5):
    for j in range(5):
        precision[i][j] = cm[i][j] / tp_fp[j]

print("recall:")
print(np.around(recall, 3))

print("precision:")
print(np.around(precision, 3))

f1_matrix = (2*recall*precision)/(recall+precision+1e-10)

print("f1:")
print(np.around(f1_matrix, 3))

#定义类别标签
labels = ['W', 'N1', 'N2', 'N3', 'REM']

#绘制Confusion Matrix矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# 绘制 Recall 矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(recall, annot=True, fmt=".2f", cmap="Greens", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Recall Matrix")
plt.show()

# 绘制 Precision 矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(precision, annot=True, fmt=".2f", cmap="Oranges", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Precision Matrix")
plt.show()

# 绘制 F1-score 矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(f1_matrix, annot=True, fmt=".2f", cmap="Purples", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("F1-score Matrix")
plt.show()
