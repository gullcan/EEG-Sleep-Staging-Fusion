import argparse
import os

import numpy as np
import torch
import sklearn.metrics as skmetrics
import matplotlib.pyplot as plt
import seaborn as sns
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


labels = ["W", "N1", "N2", "N3", "REM"]


parser = argparse.ArgumentParser(description="Test Raw EEG + Manual Feature Fusion Model")

parser.add_argument("--npz_dir", type=str, default="./data/sleepedf/npz")
parser.add_argument("--manual_dir", type=str, default="./data/sleepedf/manual_features_e100")
parser.add_argument("--model_path", type=str, default="./models/model_FusionBiLSTM_seq20_e100.pt")

parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--seq_len", type=int, default=20)

parser.add_argument("--manual_proj_dim", type=int, default=128)
parser.add_argument("--lstm_hidden_dim", type=int, default=128)

parser.add_argument("--bidirectional", action="store_true")
parser.add_argument("--num_workers", type=int, default=0)

args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Raw dosyaları bul
raw_files = scan_npz_files(args.npz_dir)

# Sadece manual feature dosyası mevcut olan raw dosyaları kullan
paired_raw_files = []

for raw_file in raw_files:
    base_name = os.path.splitext(os.path.basename(raw_file))[0]
    manual_file = os.path.join(args.manual_dir, f"{base_name}_manual.npz")

    if os.path.exists(manual_file):
        paired_raw_files.append(raw_file)

paired_raw_files = sorted(paired_raw_files)

if len(paired_raw_files) == 0:
    raise RuntimeError(
        "No paired raw/manual files found. "
        "Generate manual feature cache before testing fusion model."
    )

print(f"Raw files found          : {len(raw_files)}")
print(f"Paired raw/manual files  : {len(paired_raw_files)}")
print(f"Manual feature directory : {args.manual_dir}")


# Subject-independent split
subject_to_files = group_files_by_subject(paired_raw_files)
subject_ids = sorted(subject_to_files.keys())

print(f"Fusion subjects found: {len(subject_ids)}")

train_subjects, val_subjects, test_subjects = split_subjects(
    subject_ids=subject_ids,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
)

check_subject_overlap(train_subjects, val_subjects, test_subjects)

test_files = collect_files_for_subjects(subject_to_files, test_subjects)

print("\nFusion test split:")
print(f"Test subjects: {len(test_subjects)}")
print(f"Test files   : {len(test_files)}")


test_dataset = LazyFusionSequenceDataset(
    raw_files=test_files,
    manual_dir=args.manual_dir,
    seq_len=args.seq_len,
    debug=False
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers
)

print(f"Test sequences: {len(test_dataset)}")


# Model
model = FusionSleepNet(
    seq_len=args.seq_len,
    manual_dim=35,
    manual_proj_dim=args.manual_proj_dim,
    lstm_hidden_dim=args.lstm_hidden_dim,
    is_bidirectional=args.bidirectional
)

model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()


preds = []
trues = []


with torch.no_grad():
    for batch_idx, (X_raw, X_manual, y) in enumerate(test_dataloader):
        # X_raw shape    : [batch_size, seq_len, 3000]
        # X_manual shape : [batch_size, seq_len, 35]
        # y shape        : [batch_size, seq_len]

        X_raw = X_raw.to(device)
        X_manual = X_manual.to(device)
        y = y.reshape(-1,).to(device, dtype=torch.long)

        # pred shape: [batch_size * seq_len, 5]
        pred = model(X_raw, X_manual)

        preds.append(pred.argmax(dim=1).cpu())
        trues.append(y.cpu())


preds = np.hstack(preds)
trues = np.hstack(trues)


acc = skmetrics.accuracy_score(trues, preds)
macro_f1 = skmetrics.f1_score(trues, preds, average="macro")
kappa = skmetrics.cohen_kappa_score(trues, preds)

cm = skmetrics.confusion_matrix(
    y_true=trues,
    y_pred=preds,
    labels=[0, 1, 2, 3, 4]
)


print("\nFusion Test Results")
print(f"test_acc   : {acc * 100:.2f}%")
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
print(cm.astype("i"))


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
    print(f"True N1 predicted as {stage_name:>3}: {count:>6} | {ratio * 100:6.2f}%")


os.makedirs("./figures", exist_ok=True)

model_name = "FusionBiLSTM" if args.bidirectional else "FusionLSTM"

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix - {model_name}")
plt.tight_layout()
plt.savefig(
    f"./figures/confusion_matrix_{model_name}_seq{args.seq_len}_e100.pdf",
    dpi=300,
    bbox_inches="tight"
)
plt.show()