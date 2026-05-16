import argparse
import os
import timeit

import numpy as np
import torch
import sklearn.metrics as skmetrics
from torch.utils.data import DataLoader

from dataset import (
    scan_npz_files,
    group_files_by_subject,
    split_subjects,
    collect_files_for_subjects,
    check_subject_overlap,
    LazyFusionSequenceDataset
)
from network import FusionSleepNet
from focal_loss import FocalLoss


# 命令行传参
parser = argparse.ArgumentParser(description="Train Raw EEG + Manual Feature Fusion Model")

parser.add_argument("--npz_dir", type=str, default="./data/sleepedf/npz")
parser.add_argument("--manual_dir", type=str, default="./data/sleepedf/manual_features_e100")

parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--seq_len", type=int, default=20)

parser.add_argument("--manual_proj_dim", type=int, default=128)
parser.add_argument("--lstm_hidden_dim", type=int, default=128)

parser.add_argument("--bidirectional", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--num_workers", type=int, default=0)

args = parser.parse_args()


#定义超参数
npz_dir = args.npz_dir
manual_dir = args.manual_dir
n_epochs = args.n_epochs
batch_size = args.batch_size
seq_len = args.seq_len
manual_proj_dim = args.manual_proj_dim
lstm_hidden_dim = args.lstm_hidden_dim
bidirectional = args.bidirectional
learning_rate = 0.001


#设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


#加载 raw EEG 文件
raw_files = scan_npz_files(npz_dir)

# Sadece manual feature cache'i mevcut olan raw dosyaları kullan
paired_raw_files = []
for raw_file in raw_files:
    base_name = os.path.splitext(os.path.basename(raw_file))[0]
    manual_file = os.path.join(manual_dir, f"{base_name}_manual.npz")

    if os.path.exists(manual_file):
        paired_raw_files.append(raw_file)

paired_raw_files = sorted(paired_raw_files)

if len(paired_raw_files) == 0:
    raise RuntimeError(
        "No paired raw/manual files found. "
        "Generate manual feature cache before training fusion model."
    )

print(f"Raw files found          : {len(raw_files)}")
print(f"Paired raw/manual files  : {len(paired_raw_files)}")
print(f"Manual feature directory : {manual_dir}")


# Subject-independent split
subject_to_files = group_files_by_subject(paired_raw_files)
subject_ids = sorted(subject_to_files.keys())

print(f"Fusion subjects found: {len(subject_ids)}")

if len(subject_ids) < 3:
    raise RuntimeError(
        "Not enough subjects for subject-independent train/validation/test split."
    )

train_subjects, val_subjects, test_subjects = split_subjects(
    subject_ids=subject_ids,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
)

check_subject_overlap(train_subjects, val_subjects, test_subjects)

train_files = collect_files_for_subjects(subject_to_files, train_subjects)
val_files = collect_files_for_subjects(subject_to_files, val_subjects)
test_files = collect_files_for_subjects(subject_to_files, test_subjects)

print("\nFusion split summary:")
print(f"Train subjects      : {len(train_subjects)}")
print(f"Validation subjects : {len(val_subjects)}")
print(f"Test subjects       : {len(test_subjects)}")

print(f"Train files         : {len(train_files)}")
print(f"Validation files    : {len(val_files)}")
print(f"Test files          : {len(test_files)}")


# Dataset
train_dataset = LazyFusionSequenceDataset(
    raw_files=train_files,
    manual_dir=manual_dir,
    seq_len=seq_len,
    debug=args.debug
)

validate_dataset = LazyFusionSequenceDataset(
    raw_files=val_files,
    manual_dir=manual_dir,
    seq_len=seq_len,
    debug=args.debug
)

test_dataset = LazyFusionSequenceDataset(
    raw_files=test_files,
    manual_dir=manual_dir,
    seq_len=seq_len,
    debug=args.debug
)

print("\nFusion sequence counts:")
print(f"Train sequences      : {len(train_dataset)}")
print(f"Validation sequences : {len(validate_dataset)}")
print(f"Test sequences       : {len(test_dataset)}")


train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=args.num_workers
)

validate_dataloader = DataLoader(
    validate_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=args.num_workers
)


# Shape kontrolü
X_raw_check, X_manual_check, y_check = next(iter(train_dataloader))

print(f"Check X_raw shape    : {X_raw_check.shape}")
print(f"Check X_manual shape : {X_manual_check.shape}")
print(f"Check y shape        : {y_check.shape}")


#加载模型
model = FusionSleepNet(
    seq_len=seq_len,
    manual_dim=35,
    manual_proj_dim=manual_proj_dim,
    lstm_hidden_dim=lstm_hidden_dim,
    is_bidirectional=bidirectional
)

model.to(device)


#设置优化器
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4
)


#设置损失函数
criterion = FocalLoss(gamma=2.0)


# 学习率调度器，监控验证集loss
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.8,
    patience=5
)


os.makedirs("./models", exist_ok=True)

model_name = "FusionBiLSTM" if bidirectional else "FusionLSTM"
model_path = f"./models/model_{model_name}_seq{seq_len}_e100.pt"

print(f"Model will be saved to: {model_path}")


best_acc = -1


#训练模型
for epoch_idx in range(n_epochs):
    start = timeit.default_timer()

    model.train()
    train_loss = []
    train_trues = []
    train_preds = []

    for batch_idx, (X_raw, X_manual, y) in enumerate(train_dataloader):
        optimizer.zero_grad()

        # X_raw shape    : [batch_size, seq_len, 3000]
        # X_manual shape : [batch_size, seq_len, 35]
        # y shape        : [batch_size, seq_len]

        X_raw = X_raw.to(device)
        X_manual = X_manual.to(device)
        y = y.reshape(-1,).to(device, dtype=torch.long)

        # pred shape: [batch_size * seq_len, 5]
        pred = model(X_raw, X_manual)

        loss = criterion(pred, y)

        train_trues.append(y.cpu())
        train_preds.append(pred.argmax(dim=1).cpu())
        train_loss.append(loss.item())

        loss.backward()
        optimizer.step()

    epoch_train_loss = np.sum(train_loss)

    train_trues = np.hstack(train_trues)
    train_preds = np.hstack(train_preds)

    train_acc = skmetrics.accuracy_score(
        y_true=train_trues,
        y_pred=train_preds
    )

    train_f1_score = skmetrics.f1_score(
        train_trues,
        train_preds,
        average="macro"
    )


    #验证模型
    model.eval()
    validate_loss = []
    validate_trues = []
    validate_preds = []

    with torch.no_grad():
        for batch_idx, (X_raw, X_manual, y) in enumerate(validate_dataloader):
            X_raw = X_raw.to(device)
            X_manual = X_manual.to(device)
            y = y.reshape(-1,).to(device, dtype=torch.long)

            pred = model(X_raw, X_manual)
            loss = criterion(pred, y)

            validate_trues.append(y.cpu())
            validate_preds.append(pred.argmax(dim=1).cpu())
            validate_loss.append(loss.item())

    epoch_val_loss = np.sum(validate_loss)

    validate_trues = np.hstack(validate_trues)
    validate_preds = np.hstack(validate_preds)

    validate_acc = skmetrics.accuracy_score(
        y_true=validate_trues,
        y_pred=validate_preds
    )

    validate_f1_score = skmetrics.f1_score(
        validate_trues,
        validate_preds,
        average="macro"
    )

    end = timeit.default_timer()

    print(
        f"[epoch: {epoch_idx+1:3}/{n_epochs:3}] "
        f"|| train_loss:{epoch_train_loss:6.2f} "
        f"|| train_acc:{train_acc*100:5.2f}% "
        f"|| train_mf1:{train_f1_score:4.2f} "
        f"|| val_loss:{epoch_val_loss:6.2f} "
        f"|| val_acc:{validate_acc*100:5.2f}% "
        f"|| val_mf1:{validate_f1_score:4.2f} "
        f"({end-start:4.2f}s)"
    )

    scheduler.step(epoch_val_loss)

    if best_acc < validate_acc:
        best_acc = validate_acc
        print(f"[epoch: {epoch_idx+1:3}/{n_epochs:3}] save best model to {model_path}...")
        torch.save(model.state_dict(), model_path)