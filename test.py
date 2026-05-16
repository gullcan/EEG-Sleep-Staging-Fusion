import torch

from network import SleepNet
from dataset import create_dataloaders

import sklearn.metrics as skmetrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import argparse


#定义类别标签
labels = ['W', 'N1', 'N2', 'N3', 'REM']


# 命令行传参
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seq_len", type=int, default=64)
parser.add_argument("--network", type=str, default="GRU", choices=["GRU", "LSTM", "Attention"], help="GRU | LSTM | Attention")
parser.add_argument("--bidirectional", action="store_true")

# 要注意这里训练得到的模型要与训练时选用的网络结构相对于，就你用GRU跑的，那自然测试的时候也是用GRU
parser.add_argument("--model_path", type=str, default="./models/model_GRU_seq20.pt", help="model path")
parser.add_argument("--data_path", type=str, default="./data/sleepedf/npz", help="data path")
parser.add_argument("--num_workers", type=int, default=0)
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 加载模型
model = SleepNet(
    network=args.network,
    seq_len=args.seq_len,
    is_bidirectional=args.bidirectional
)

model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()


# 加载数据集
# Burada train ve validation loader da oluşturulur,
# fakat test değerlendirmesi sadece test_dataloader üzerinde yapılır.
train_dataloader, validate_dataloader, test_dataloader = create_dataloaders(
    npz_dir=args.data_path,
    seq_len=args.seq_len,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    debug=False
)


preds = []
trues = []


# 测试模型
with torch.no_grad():
    for batch_idx, (X, y) in enumerate(test_dataloader):
        # X shape: [batch_size, seq_len, 3000]
        # y shape: [batch_size, seq_len]
        X = X.to(device)
        y = y.reshape(-1,).to(device, dtype=torch.long)

        # pred shape: [batch_size * seq_len, 5]
        pred = model(X)

        preds.append(pred.argmax(dim=1).cpu())
        trues.append(y.cpu())


preds = np.hstack(preds)
trues = np.hstack(trues)


# 基本评价指标
acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
macro_f1 = skmetrics.f1_score(trues, preds, average="macro")
kappa = skmetrics.cohen_kappa_score(trues, preds)

cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0, 1, 2, 3, 4])


print("\nTest results")
print(f"test_acc   : {acc*100:.2f}%")
print(f"test_mf1   : {macro_f1:.4f}")
print(f"test_kappa : {kappa:.4f}")

print("\nclassification_report:")
print(
    skmetrics.classification_report(
        trues,
        preds,
        labels=[0, 1, 2, 3, 4],
        target_names=labels,
        digits=4
    )
)

print("confusion_matrix:")
print(cm.astype('i'))


# sklearn计算的混淆矩阵的列为预测值，行为真实值

# tp_fp 预测值, 列和
tp_fp = cm.sum(axis=0)  # 每一列求和：预测值统计
# tp_fn 真实值, 行和
tp_fn = cm.sum(axis=1)  # 每一列求和：预测值统计


print("tp_fp", tp_fp)
print("tp_fn", tp_fn)


precision = np.zeros((5, 5))
recall = np.zeros((5, 5))


for i in range(5):
    if tp_fn[i] != 0:
        recall[i] = cm[i] / tp_fn[i]


for i in range(5):
    for j in range(5):
        if tp_fp[j] != 0:
            precision[i][j] = cm[i][j] / tp_fp[j]


print("recall:")
print(np.around(recall, 3))


print("precision:")
print(np.around(precision, 3))


f1_matrix = (2 * recall * precision) / (recall + precision + 1e-10)


print("f1:")
print(np.around(f1_matrix, 3))


# N1 özel hata analizi
n1_true_index = 1
n1_total = cm[n1_true_index].sum()
n1_correct = cm[n1_true_index, n1_true_index]
n1_recall = n1_correct / n1_total if n1_total > 0 else 0.0

print("\nN1 specific error analysis:")
print(f"N1 total true samples : {n1_total}")
print(f"N1 correctly predicted: {n1_correct}")
print(f"N1 recall             : {n1_recall:.4f}")

for pred_idx, stage_name in enumerate(labels):
    count = cm[n1_true_index, pred_idx]
    ratio = count / n1_total if n1_total > 0 else 0.0
    print(f"True N1 predicted as {stage_name:>3}: {count:>6} | {ratio*100:6.2f}%")


# 绘制Confusion Matrix矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(f"confusion_matrix_{args.network}_seq{args.seq_len}.pdf", dpi=300, bbox_inches="tight")
plt.show()


# 绘制 Recall 矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(recall, annot=True, fmt=".2f", cmap="Greens", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Recall Matrix")
plt.tight_layout()
plt.savefig(f"recall_matrix_{args.network}_seq{args.seq_len}.pdf", dpi=300, bbox_inches="tight")
plt.show()


# 绘制 Precision 矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(precision, annot=True, fmt=".2f", cmap="Oranges", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Precision Matrix")
plt.tight_layout()
plt.savefig(f"precision_matrix_{args.network}_seq{args.seq_len}.pdf", dpi=300, bbox_inches="tight")
plt.show()


# 绘制 F1-score 矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(f1_matrix, annot=True, fmt=".2f", cmap="Purples", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("F1-score Matrix")
plt.tight_layout()
plt.savefig(f"f1_matrix_{args.network}_seq{args.seq_len}.pdf", dpi=300, bbox_inches="tight")
plt.show()