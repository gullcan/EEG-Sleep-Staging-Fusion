from torch.utils.data import DataLoader,random_split
import torch.nn as nn
import torch

import sklearn.metrics as skmetrics
import numpy as np
import timeit
import matplotlib.pyplot as plt
import seaborn as sns


from dataset import EdfDataset
from network import SleepNet
from focal_loss import FocalLoss

import argparse

# wandb用于在线追溯实验，方便实验结果保存和调参
# import wandb

#设置seaborn样式
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# 命令行传参
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seq_len", type=int, default=64)
parser.add_argument("--network", type=str, default="GRU", help="GRU | LSTM | Attention")
args = parser.parse_args()

#定义超参数
n_epochs = args.n_epochs # 迭代次数,每个epoch会对整个训练集遍历一遍
batch_size = args.batch_size # 一次加载的数据量，对一个epoch中的样本数的拆分
learning_rate = 0.001 # 学习率
seq_len = args.seq_len
network = args.network


#加载数据
data_path = "./data/sleepedf/npz"
train_data = EdfDataset(data_path, seq_len=seq_len, is_train=False)
#按8:2划分训练集和验证集
train_size = int(len(train_data)*0.8)
validate_size = len(train_data) - train_size
train_dataset,validate_dataset = random_split(train_data,[train_size,validate_size])
#使用DataLoader加载数据集，转换为迭代器
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
validate_dataloader = DataLoader(validate_dataset,batch_size=batch_size) #用于训练中验证模型效果，进而可以动态调整超参数，控制训练

#设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

#加载模型
model = SleepNet(network=network, seq_len=seq_len)
# model.load_state_dict(torch.load("./models/model_GRU.pt"))
model.to(device)

#设置优化器
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4)

#设置损失函数
criterion = FocalLoss(gamma=2.0)

# 学习率调度器，监控验证集loss
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=5, verbose=True)

best_acc = -1

#用于记录每个epoch的损失，用于绘图
train_losses_all = []
val_losses_all = []
train_acc_all = []
val_acc_all = []
train_f1_all = []
val_f1_all = []

#训练模型
for epoch_idx in range(n_epochs):
    start = timeit.default_timer()
    model.train()
    train_loss = []
    train_trues = []
    train_preds = []
    for batch_idx, (X, y) in enumerate(train_dataloader):
        optimizer.zero_grad()
        X, y = X.to(device), y.reshape(-1,).to(device, dtype=torch.long)
        pred = model(X)
        loss = criterion(pred, y)
        train_trues.append(y.cpu())
        train_preds.append(pred.argmax(dim=1).cpu())
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    #计算训练集整体loss和准确率
    epoch_train_loss = np.sum(train_loss)
    train_losses_all.append(epoch_train_loss)
    train_trues = np.hstack(train_trues)
    train_preds = np.hstack(train_preds)
    train_acc = skmetrics.accuracy_score(y_true=train_trues, y_pred=train_preds)
    train_f1_score = skmetrics.f1_score(train_trues,train_preds,average="macro")

    #验证模型
    model.eval()
    validate_loss = []
    validate_trues = []
    validate_preds = []
    with torch.no_grad(): #不计算梯度，加快运算速度
        for batch_idx, (X, y) in enumerate(validate_dataloader):
            X, y = X.to(device), y.reshape(-1,).to(device, dtype=torch.long)
            pred = model(X)
            loss = criterion(pred, y)
            validate_trues.append(y.cpu())
            validate_preds.append(pred.argmax(dim=1).cpu())
            validate_loss.append(loss.item())

    epoch_val_loss = np.sum(validate_loss)
    val_losses_all.append(epoch_val_loss)
    validate_trues = np.hstack(validate_trues)
    validate_preds = np.hstack(validate_preds)
    validate_acc = skmetrics.accuracy_score(y_true=validate_trues, y_pred=validate_preds)
    validate_f1_score = skmetrics.f1_score(validate_trues,validate_preds,average="macro")

    end = timeit.default_timer()        
    
    print(f"[epoch: {epoch_idx+1:3}/{n_epochs:3}] || train_loss:{np.sum(train_loss):6.2f} || train_acc:{train_acc*100:5.2f}% || train_mf1:{train_f1_score:4.2f}\
    || val_loss:{np.sum(validate_loss):6.2f} || val_acc:{validate_acc*100:5.2f}% || val_mf1:{validate_f1_score:4.2f} ({end-start:4.2f}s)")

    # 记录每个 epoch 的 acc 和 f1 分数
    train_acc_all.append(train_acc)
    train_f1_all.append(train_f1_score)
    val_acc_all.append(validate_acc)
    val_f1_all.append(validate_f1_score)

    # 更新学习率调度器
    scheduler.step(epoch_val_loss)


    if best_acc < validate_acc:
        best_acc = validate_acc
        print(f"[epoch: {epoch_idx+1:3}/{n_epochs:3}] save best model...")
        torch.save(model.state_dict(), "./models/model_GRU.pt")
parser = argparse.ArgumentParser(description="Train SleepNet Model")

parser.add_argument("--n_epochs", type=int, default=150)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--seq_len", type=int, default=64)
parser.add_argument("--network", type=str, default="GRU", choices=["GRU", "LSTM", "Attention"])
args = parser.parse_args()

# 平滑曲线：使用 SciPy spline 插值
from scipy.interpolate import make_interp_spline
def smooth_curve(x, y, num_points=300, k=3):
    x = np.array(x)
    y = np.array(y)
    spline = make_interp_spline(x, y, k=k)
    x_smooth = np.linspace(x.min(), x.max(), num_points)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

# Loss 曲线
epochs_arr = np.array(range(1, len(train_losses_all) + 1))
x_loss_smooth, smooth_train_losses = smooth_curve(epochs_arr, train_losses_all)
_, smooth_val_losses = smooth_curve(epochs_arr, val_losses_all)

plt.figure(figsize=(10, 6))
plt.plot(x_loss_smooth, smooth_train_losses, label='Training Loss', color='blue', linewidth=2)
plt.plot(x_loss_smooth, smooth_val_losses, label='Validation Loss', color='red', linestyle='--', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title(f'Smoothed Training and Validation Loss ({network})')
plt.legend(loc="best")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f'loss_curve_{network}.pdf', dpi=300, bbox_inches='tight')
plt.show()

# Accuracy 曲线
x_acc_smooth, smooth_train_acc = smooth_curve(epochs_arr, np.array(train_acc_all)*100)
_, smooth_val_acc = smooth_curve(epochs_arr, np.array(val_acc_all)*100)

plt.figure(figsize=(10, 6))
plt.plot(x_acc_smooth, smooth_train_acc, label='Training Accuracy', color='green', marker='', linestyle='-', linewidth=2)
plt.plot(x_acc_smooth, smooth_val_acc, label='Validation Accuracy', color='orange', marker='', linestyle='--', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title(f'Smoothed Training & Validation Accuracy ({network})')
plt.legend(loc="best")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f'acc_curve_{network}.pdf', dpi=300, bbox_inches='tight')
plt.show()

# F1-score 曲线
x_f1_smooth, smooth_train_f1 = smooth_curve(epochs_arr, train_f1_all)
_, smooth_val_f1 = smooth_curve(epochs_arr, val_f1_all)

plt.figure(figsize=(10, 6))
plt.plot(x_f1_smooth, smooth_train_f1, label='Training Macro F1', color='purple', marker='', linestyle='-', linewidth=2)
plt.plot(x_f1_smooth, smooth_val_f1, label='Validation Macro F1', color='brown', marker='', linestyle='--', linewidth=2)
plt.xlabel('Epochs')
plt.ylabel('Macro F1 Score')
plt.title(f'Smoothed Training & Validation F1 Score ({network})')
plt.legend(loc="best")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(f'f1_curve_{network}.pdf', dpi=300, bbox_inches='tight')
plt.show()



#     wandb.log({"train_loss": np.sum(train_loss), 
#                "train_acc": train_acc,
#                "val_loss": np.sum(validate_loss),
#                "val_acc": validate_acc})

# wandb.finish()