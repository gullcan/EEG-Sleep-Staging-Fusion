import argparse
import os

import numpy as np
import torch
import sklearn.metrics as skmetrics
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader

from dataset import (
    scan_manual_feature_files,
    LazyManualFeatureSequenceDataset,
    group_files_by_subject,
    split_subjects,
    collect_files_for_subjects,
    check_subject_overlap
)
from network import ManualFeatureNet


labels = ["W", "N1", "N2", "N3", "REM"]


parser = argparse.ArgumentParser(description="Test Manual Feature Sleep Staging Model")
parser.add_argument("--manual_dir", type=str, default="./data/sleepedf/manual_features_e100")
parser.add_argument("--model_path", type=str, default="./models/model_ManualBiLSTM_seq20_e100.pt")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--seq_len", type=int, default=20)
parser.add_argument("--hidden_dim", type=int, default=128)
parser.add_argument("--bidirectional", action="store_true")
parser.add_argument("--num_workers", type=int, default=0)
args = parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


files = scan_manual_feature_files(args.manual_dir)
print(f"Manual feature files found: {len(files)}")
print(f"Manual feature directory: {args.manual_dir}")


subject_to_files = group_files_by_subject(files)
subject_ids = sorted(subject_to_files.keys())

print(f"Manual feature subjects found: {len(subject_ids)}")

train_subjects, val_subjects, test_subjects = split_subjects(
    subject_ids=subject_ids,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
)

check_subject_overlap(train_subjects, val_subjects, test_subjects)

test_files = collect_files_for_subjects(subject_to_files, test_subjects)

print("\nManual feature test split:")
print(f"Test subjects: {len(test_subjects)}")
print(f"Test files   : {len(test_files)}")


test_dataset = LazyManualFeatureSequenceDataset(
    files=test_files,
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


model = ManualFeatureNet(
    input_dim=35,
    hidden_dim=args.hidden_dim,
    seq_len=args.seq_len,
    is_bidirectional=args.bidirectional
)

model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()


preds = []
trues = []


with torch.no_grad():
    for batch_idx, (X, y) in enumerate(test_dataloader):
        # X shape: [batch_size, seq_len, 35]
        # y shape: [batch_size, seq_len]
        X = X.to(device)
        y = y.reshape(-1,).to(device, dtype=torch.long)

        # pred shape: [batch_size * seq_len, 5]
        pred = model(X)

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


print("\nManual Feature Test Results")
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

model_name = "ManualBiLSTM" if args.bidirectional else "ManualLSTM"

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
plt.savefig(f"./figures/confusion_matrix_{model_name}_seq{args.seq_len}_e100.pdf", dpi=300, bbox_inches="tight")
plt.show()