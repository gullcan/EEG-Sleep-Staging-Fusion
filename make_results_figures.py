import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)


# ============================================================
# Full subject-independent test results
# ============================================================

results = [
    {
        "Model": "CNN-BiLSTM",
        "Raw_EEG": "Yes",
        "Manual_Features": "No",
        "Attention": "No",
        "Accuracy": 78.34,
        "Macro_F1": 0.6574,
        "Kappa": 0.6910,
        "W_F1": 0.8961,
        "N1_F1": 0.1758,
        "N2_F1": 0.8178,
        "N3_F1": 0.7599,
        "REM_F1": 0.6372,
        "N1_Recall": 0.1049,
    },
    {
        "Model": "ManualFeature-BiLSTM",
        "Raw_EEG": "No",
        "Manual_Features": "Yes",
        "Attention": "No",
        "Accuracy": 75.62,
        "Macro_F1": 0.6531,
        "Kappa": 0.6555,
        "W_F1": 0.8785,
        "N1_F1": 0.2686,
        "N2_F1": 0.7809,
        "N3_F1": 0.7526,
        "REM_F1": 0.5850,
        "N1_Recall": 0.1780,
    },
    {
        "Model": "FusionBiLSTM",
        "Raw_EEG": "Yes",
        "Manual_Features": "Yes",
        "Attention": "No",
        "Accuracy": 79.70,
        "Macro_F1": 0.7128,
        "Kappa": 0.7142,
        "W_F1": 0.9179,
        "N1_F1": 0.4174,
        "N2_F1": 0.8238,
        "N3_F1": 0.7320,
        "REM_F1": 0.6732,
        "N1_Recall": 0.3910,
    },
    {
        "Model": "FusionAttention-BiLSTM",
        "Raw_EEG": "Yes",
        "Manual_Features": "Yes",
        "Attention": "Yes",
        "Accuracy": 78.36,
        "Macro_F1": 0.6929,
        "Kappa": 0.6964,
        "W_F1": 0.8906,
        "N1_F1": 0.3309,
        "N2_F1": 0.8020,
        "N3_F1": 0.7601,
        "REM_F1": 0.6810,
        "N1_Recall": 0.2516,
    },
]


df = pd.DataFrame(results)

csv_path = "results/results_summary.csv"
df.to_csv(csv_path, index=False)

print(f"Saved results table to: {csv_path}")
print(df)


# ============================================================
# Figure 1: Overall model comparison
# Accuracy, Macro F1, Kappa
# ============================================================

models = df["Model"].tolist()
x = np.arange(len(models))
width = 0.25

accuracy = df["Accuracy"].values / 100.0
macro_f1 = df["Macro_F1"].values
kappa = df["Kappa"].values

plt.figure(figsize=(11, 6))

plt.bar(x - width, accuracy, width, label="Accuracy")
plt.bar(x, macro_f1, width, label="Macro F1")
plt.bar(x + width, kappa, width, label="Cohen's Kappa")

plt.xticks(x, models, rotation=20, ha="right")
plt.ylabel("Score")
plt.ylim(0, 1.0)
plt.title("Overall Model Performance Comparison")
plt.legend()
plt.tight_layout()

figure_path = "figures/model_performance_comparison.pdf"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.savefig("figures/model_performance_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {figure_path}")


# ============================================================
# Figure 2: Per-class F1-score comparison
# ============================================================

class_columns = ["W_F1", "N1_F1", "N2_F1", "N3_F1", "REM_F1"]
class_labels = ["W", "N1", "N2", "N3", "REM"]

x = np.arange(len(class_labels))
width = 0.18

plt.figure(figsize=(12, 6))

for i, (_, row) in enumerate(df.iterrows()):
    values = row[class_columns].values.astype(float)
    plt.bar(x + (i - 1.5) * width, values, width, label=row["Model"])

plt.xticks(x, class_labels)
plt.ylabel("F1-score")
plt.ylim(0, 1.0)
plt.title("Per-Class F1-score Comparison")
plt.legend()
plt.tight_layout()

figure_path = "figures/per_class_f1_comparison.pdf"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.savefig("figures/per_class_f1_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {figure_path}")


# ============================================================
# Figure 3: N1 recall comparison
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(df["Model"], df["N1_Recall"])
plt.ylabel("N1 Recall")
plt.ylim(0, 0.5)
plt.title("N1 Recall Comparison")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

figure_path = "figures/n1_recall_comparison.pdf"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.savefig("figures/n1_recall_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {figure_path}")


# ============================================================
# Figure 4: N1 F1-score comparison
# ============================================================

plt.figure(figsize=(9, 5))

plt.bar(df["Model"], df["N1_F1"])
plt.ylabel("N1 F1-score")
plt.ylim(0, 0.5)
plt.title("N1 F1-score Comparison")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

figure_path = "figures/n1_f1_comparison.pdf"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.savefig("figures/n1_f1_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved figure to: {figure_path}")


print("\nAll result tables and figures were created successfully.")