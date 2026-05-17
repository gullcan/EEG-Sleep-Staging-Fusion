import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report

from dataset import (
    scan_manual_feature_files,
    group_files_by_subject,
    split_subjects,
    collect_files_for_subjects,
    check_subject_overlap
)


STAGE_LABELS = ["W", "N1", "N2", "N3", "REM"]


SIGNAL_NAMES = [
    "original",
    "delta",
    "theta",
    "alpha",
    "beta"
]


FEATURE_TYPES = [
    "energy",
    "approximate_entropy",
    "fuzzy_entropy",
    "multiscale_entropy",
    "permutation_entropy",
    "power_spectral_entropy",
    "envelope_entropy"
]


def get_manual_feature_names():
    """
    35-dimensional manual feature vektörü için özellik isimleri üretir.

    Order:
        original_*
        delta_*
        theta_*
        alpha_*
        beta_*
    """
    names = []

    for signal_name in SIGNAL_NAMES:
        for feature_type in FEATURE_TYPES:
            names.append(f"{signal_name}_{feature_type}")

    if len(names) != 35:
        raise RuntimeError(f"Expected 35 feature names, got {len(names)}")

    return names


def load_manual_epoch_data(files):
    """
    Manual feature .npz dosyalarını epoch-level matrix olarak yükler.

    Output:
        X shape: [num_epochs, 35]
        y shape: [num_epochs]
    """
    X_list = []
    y_list = []

    for file_path in files:
        data = np.load(file_path)

        x_manual = data["x_manual"]
        y = data["y"]

        if x_manual.shape[1] != 35:
            raise RuntimeError(f"Expected 35 features, got {x_manual.shape[1]} in {file_path}")

        if x_manual.shape[0] != len(y):
            raise RuntimeError(f"x/y mismatch in {file_path}")

        X_list.append(x_manual.astype(np.float32))
        y_list.append(y.astype(np.int64))

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    return X, y


def limit_samples(X, y, max_samples, seed=42):
    """
    Hızlı deneme için örnek sayısını sınırlar.
    max_samples=None ise tüm veriyi kullanır.
    """
    if max_samples is None:
        return X, y

    if len(y) <= max_samples:
        return X, y

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(y), size=max_samples, replace=False)

    return X[indices], y[indices]


def train_random_forest_for_importance(X_train, y_train, n_estimators=300, seed=42):
    """
    Random Forest ile feature importance hesaplar.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    model.fit(X_train, y_train)

    return model


def evaluate_selected_features(X_train, y_train, X_test, y_test, selected_indices, n_estimators=300, seed=42):
    """
    Seçilmiş feature subset'i ile Random Forest eğitip test eder.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample"
    )

    clf.fit(X_train[:, selected_indices], y_train)

    preds = clf.predict(X_test[:, selected_indices])

    acc = accuracy_score(y_test, preds)
    mf1 = f1_score(y_test, preds, average="macro")
    kappa = cohen_kappa_score(y_test, preds)

    report = classification_report(
        y_test,
        preds,
        labels=[0, 1, 2, 3, 4],
        target_names=STAGE_LABELS,
        digits=4,
        output_dict=True
    )

    return {
        "accuracy": acc,
        "macro_f1": mf1,
        "kappa": kappa,
        "W_F1": report["W"]["f1-score"],
        "N1_F1": report["N1"]["f1-score"],
        "N2_F1": report["N2"]["f1-score"],
        "N3_F1": report["N3"]["f1-score"],
        "REM_F1": report["REM"]["f1-score"],
        "N1_recall": report["N1"]["recall"]
    }


def plot_feature_importance(feature_df, output_path_png, output_path_pdf, top_n=20):
    """
    En önemli feature'ları bar chart olarak kaydeder.
    """
    top_df = feature_df.head(top_n).iloc[::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(top_df["feature_name"], top_df["importance"])
    plt.xlabel("Random Forest Feature Importance")
    plt.ylabel("Manual Feature")
    plt.title(f"Top {top_n} Manual EEG Features")
    plt.tight_layout()
    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_path_pdf, dpi=300, bbox_inches="tight")
    plt.show()


def plot_topk_results(results_df, output_path_png, output_path_pdf):
    """
    Top-k feature selection sonuçlarını çizer.
    """
    plt.figure(figsize=(9, 5))

    x = np.arange(len(results_df))
    width = 0.25

    plt.bar(x - width, results_df["accuracy"], width, label="Accuracy")
    plt.bar(x, results_df["macro_f1"], width, label="Macro F1")
    plt.bar(x + width, results_df["kappa"], width, label="Cohen's Kappa")

    plt.xticks(x, results_df["setting"])
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Top-k Manual Feature Selection Performance")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_path_pdf, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Manual EEG feature selection analysis")
    parser.add_argument("--manual_dir", type=str, default="./data/sleepedf/manual_features_e100")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    feature_names = get_manual_feature_names()

    files = scan_manual_feature_files(args.manual_dir)

    print(f"Manual feature files found: {len(files)}")
    print(f"Manual feature directory   : {args.manual_dir}")

    subject_to_files = group_files_by_subject(files)
    subject_ids = sorted(subject_to_files.keys())

    print(f"Subjects found: {len(subject_ids)}")

    train_subjects, val_subjects, test_subjects = split_subjects(
        subject_ids=subject_ids,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed
    )

    check_subject_overlap(train_subjects, val_subjects, test_subjects)

    train_files = collect_files_for_subjects(subject_to_files, train_subjects)
    test_files = collect_files_for_subjects(subject_to_files, test_subjects)

    print("\nFeature selection split summary:")
    print(f"Train subjects: {len(train_subjects)}")
    print(f"Test subjects : {len(test_subjects)}")
    print(f"Train files   : {len(train_files)}")
    print(f"Test files    : {len(test_files)}")

    print("\nLoading train manual features...")
    X_train, y_train = load_manual_epoch_data(train_files)

    print("Loading test manual features...")
    X_test, y_test = load_manual_epoch_data(test_files)

    X_train, y_train = limit_samples(
        X_train,
        y_train,
        max_samples=args.max_train_samples,
        seed=args.seed
    )

    X_test, y_test = limit_samples(
        X_test,
        y_test,
        max_samples=args.max_test_samples,
        seed=args.seed
    )

    print("\nData shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"y_test : {y_test.shape}")

    print("\nTraining Random Forest for feature importance...")
    rf_model = train_random_forest_for_importance(
        X_train,
        y_train,
        n_estimators=args.n_estimators,
        seed=args.seed
    )

    importances = rf_model.feature_importances_

    feature_df = pd.DataFrame({
        "feature_index": np.arange(len(feature_names)),
        "feature_name": feature_names,
        "importance": importances
    })

    feature_df = feature_df.sort_values(
        by="importance",
        ascending=False
    ).reset_index(drop=True)

    feature_importance_path = "results/feature_importance.csv"
    feature_df.to_csv(feature_importance_path, index=False)

    print(f"\nSaved feature importance table to: {feature_importance_path}")
    print("\nTop 15 features:")
    print(feature_df.head(15))

    plot_feature_importance(
        feature_df,
        output_path_png="figures/manual_feature_importance_top20.png",
        output_path_pdf="figures/manual_feature_importance_top20.pdf",
        top_n=20
    )

    print("\nEvaluating top-k selected features...")

    topk_settings = [5, 10, 15, 25, 35]
    results = []

    for k in topk_settings:
        selected_indices = feature_df.head(k)["feature_index"].values

        metrics = evaluate_selected_features(
            X_train,
            y_train,
            X_test,
            y_test,
            selected_indices=selected_indices,
            n_estimators=args.n_estimators,
            seed=args.seed
        )

        setting_name = f"Top {k}" if k < 35 else "All 35"

        row = {
            "setting": setting_name,
            "num_features": k,
            **metrics
        }

        results.append(row)

        print(
            f"{setting_name:>6} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"MF1: {metrics['macro_f1']:.4f} | "
            f"Kappa: {metrics['kappa']:.4f} | "
            f"N1_F1: {metrics['N1_F1']:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_path = "results/feature_selection_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\nSaved top-k feature selection results to: {results_path}")
    print(results_df)

    plot_topk_results(
        results_df,
        output_path_png="figures/topk_feature_selection_comparison.png",
        output_path_pdf="figures/topk_feature_selection_comparison.pdf"
    )

    print("\nFeature selection analysis completed successfully.")


if __name__ == "__main__":
    main()