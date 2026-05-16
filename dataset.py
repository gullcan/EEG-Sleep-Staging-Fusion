import argparse
import glob
import os
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


STAGE_NAMES = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}


def get_subject_id(file_path):
    """
    Sleep-EDF dosya adından subject id çıkarır.

    Örnek:
        SC4001.npz -> SC400
        SC4002.npz -> SC400

    Böylece aynı subject'e ait farklı kayıtlar
    train / validation / test arasında karışmaz.
    """
    stem = Path(file_path).stem

    # Sleep-EDF formatı genellikle SC4001, SC4002 gibi gelir.
    # İlk 5 karakter subject id olarak kullanılır.
    return stem[:5]


def scan_npz_files(npz_dir):
    """
    Verilen klasördeki .npz dosyalarını bulur ve sıralar.
    """
    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))

    if len(files) == 0:
        raise FileNotFoundError(f"No .npz files found in: {npz_dir}")

    return files


def group_files_by_subject(files):
    """
    .npz dosyalarını subject id'ye göre gruplar.

    Çıktı:
        {
            "SC400": ["SC4001.npz", "SC4002.npz"],
            "SC401": ["SC4011.npz", "SC4012.npz"],
            ...
        }
    """
    subject_to_files = defaultdict(list)

    for file_path in files:
        subject_id = get_subject_id(file_path)
        subject_to_files[subject_id].append(file_path)

    return dict(subject_to_files)


def split_subjects(subject_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Subject-independent train / validation / test split yapar.

    Aynı subject sadece tek bir split içinde yer alır.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")

    subject_ids = np.array(sorted(subject_ids))

    rng = np.random.default_rng(seed)
    rng.shuffle(subject_ids)

    n_subjects = len(subject_ids)
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)

    train_subjects = subject_ids[:n_train]
    val_subjects = subject_ids[n_train:n_train + n_val]
    test_subjects = subject_ids[n_train + n_val:]

    return list(train_subjects), list(val_subjects), list(test_subjects)


def collect_files_for_subjects(subject_to_files, selected_subjects):
    """
    Seçilen subject'lere ait bütün dosyaları döndürür.
    """
    selected_files = []

    for subject_id in selected_subjects:
        selected_files.extend(subject_to_files[subject_id])

    return sorted(selected_files)


def check_subject_overlap(train_subjects, val_subjects, test_subjects):
    """
    Train / validation / test arasında subject overlap var mı kontrol eder.
    """
    train_set = set(train_subjects)
    val_set = set(val_subjects)
    test_set = set(test_subjects)

    train_val_overlap = train_set.intersection(val_set)
    train_test_overlap = train_set.intersection(test_set)
    val_test_overlap = val_set.intersection(test_set)

    print("\nSubject overlap check:")
    print(f"Train ∩ Validation: {len(train_val_overlap)}")
    print(f"Train ∩ Test      : {len(train_test_overlap)}")
    print(f"Validation ∩ Test : {len(val_test_overlap)}")

    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise RuntimeError("Subject overlap detected! This causes data leakage.")

    print("Subject overlap check passed. No leakage between splits.")


def compute_label_distribution(files):
    """
    Verilen .npz dosyalarındaki label dağılımını hesaplar.

    Bu fonksiyon sadece y array'ini okur.
    EEG sinyalini RAM'e topluca yüklemez.
    """
    counter = Counter()

    for file_path in files:
        data = np.load(file_path)
        labels = data["y"]
        counter.update(labels.tolist())

    total = sum(counter.values())

    return counter, total


def print_label_distribution(name, files):
    """
    Split içindeki sınıf dağılımını yazdırır.
    """
    counter, total = compute_label_distribution(files)

    print(f"\n{name} label distribution:")
    print(f"Total epochs: {total}")

    for label_id in range(5):
        count = counter.get(label_id, 0)
        ratio = 100 * count / total if total > 0 else 0
        stage_name = STAGE_NAMES[label_id]
        print(f"{stage_name:>3} ({label_id}): {count:>7} epochs | {ratio:6.2f}%")


class LazySleepSequenceDataset(Dataset):
    """
    Sleep-EDF .npz dosyalarını RAM'e tamamen yüklemeden sequence üretir.

    Her .npz dosyasında:
        x shape: [num_epochs, 3000]
        y shape: [num_epochs]

    Dataset çıktısı:
        X shape: [seq_len, 3000]
        y shape: [seq_len]

    DataLoader sonrası:
        X shape: [batch_size, seq_len, 3000]
        y shape: [batch_size, seq_len]
    """

    def __init__(self, files, seq_len=20, stride=None, debug=False, max_sequences_per_file=5):
        super().__init__()

        self.files = sorted(files)
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.debug = debug
        self.max_sequences_per_file = max_sequences_per_file

        self.index = []

        self._build_index()

    def _build_index(self):
        """
        RAM'de EEG verisi değil, sadece sequence index bilgisi tutulur.

        self.index elemanları:
            {
                "file_path": ".../SC4001.npz",
                "subject_id": "SC400",
                "start": 0
            }
        """
        for file_path in self.files:
            data = np.load(file_path)
            num_epochs = len(data["x"])

            max_start = num_epochs - self.seq_len

            if max_start < 0:
                continue

            starts = list(range(0, max_start + 1, self.stride))

            if self.debug:
                starts = starts[:self.max_sequences_per_file]

            subject_id = get_subject_id(file_path)

            for start in starts:
                self.index.append({
                    "file_path": file_path,
                    "subject_id": subject_id,
                    "start": start
                })

        if len(self.index) == 0:
            raise RuntimeError("No valid sequences were created. Check seq_len and npz files.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        item = self.index[idx]

        file_path = item["file_path"]
        start = item["start"]
        end = start + self.seq_len

        data = np.load(file_path)

        x = data["x"][start:end]
        y = data["y"][start:end]

        # Beklenen shape kontrolleri
        # x: [seq_len, 3000]
        # y: [seq_len]
        if x.shape[0] != self.seq_len:
            raise RuntimeError(f"Invalid x sequence length: {x.shape}")

        if y.shape[0] != self.seq_len:
            raise RuntimeError(f"Invalid y sequence length: {y.shape}")

        # PyTorch tensor dönüşümü
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

class LazyManualFeatureSequenceDataset(Dataset):
    """
    Manual feature .npz dosyalarını sequence formatında okur.

    Her manual .npz dosyasında:
        x_manual shape: [num_epochs, 35]
        y shape       : [num_epochs]

    Dataset çıktısı:
        X_manual shape: [seq_len, 35]
        y shape       : [seq_len]

    DataLoader sonrası:
        X_manual shape: [batch_size, seq_len, 35]
        y shape       : [batch_size, seq_len]
    """

    def __init__(self, files, seq_len=20, stride=None, debug=False, max_sequences_per_file=5):
        super().__init__()

        self.files = sorted(files)
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.debug = debug
        self.max_sequences_per_file = max_sequences_per_file

        self.index = []

        self._build_index()

    def _build_index(self):
        """
        RAM'de manual feature verisinin tamamı değil,
        sadece dosya yolu ve sequence başlangıç indexi tutulur.
        """
        for file_path in self.files:
            data = np.load(file_path)

            if "x_manual" not in data or "y" not in data:
                raise RuntimeError(f"Invalid manual feature file: {file_path}")

            x_manual = data["x_manual"]
            y = data["y"]

            if x_manual.shape[0] != len(y):
                raise RuntimeError(f"x_manual and y length mismatch in {file_path}")

            if x_manual.shape[1] != 35:
                raise RuntimeError(f"Expected 35 manual features, got {x_manual.shape[1]} in {file_path}")

            num_epochs = len(x_manual)
            max_start = num_epochs - self.seq_len

            if max_start < 0:
                continue

            starts = list(range(0, max_start + 1, self.stride))

            if self.debug:
                starts = starts[:self.max_sequences_per_file]

            subject_id = get_subject_id(file_path)

            for start in starts:
                self.index.append({
                    "file_path": file_path,
                    "subject_id": subject_id,
                    "start": start
                })

        if len(self.index) == 0:
            raise RuntimeError("No valid manual feature sequences were created.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        item = self.index[idx]

        file_path = item["file_path"]
        start = item["start"]
        end = start + self.seq_len

        data = np.load(file_path)

        x_manual = data["x_manual"][start:end]
        y = data["y"][start:end]

        if x_manual.shape != (self.seq_len, 35):
            raise RuntimeError(f"Invalid x_manual shape: {x_manual.shape}")

        if y.shape[0] != self.seq_len:
            raise RuntimeError(f"Invalid y shape: {y.shape}")

        x_manual = torch.tensor(x_manual, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x_manual, y

def scan_manual_feature_files(manual_dir):
    """
    Manual feature klasöründeki *_manual.npz dosyalarını bulur.
    """
    files = sorted(glob.glob(os.path.join(manual_dir, "*_manual.npz")))

    if len(files) == 0:
        raise FileNotFoundError(f"No manual feature files found in: {manual_dir}")

    return files

def create_datasets(
    npz_dir,
    seq_len=20,
    stride=None,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    debug=False
):
    """
    Subject-independent train / validation / test Dataset oluşturur.
    """
    files = scan_npz_files(npz_dir)
    subject_to_files = group_files_by_subject(files)

    subject_ids = sorted(subject_to_files.keys())

    train_subjects, val_subjects, test_subjects = split_subjects(
        subject_ids=subject_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    check_subject_overlap(train_subjects, val_subjects, test_subjects)

    train_files = collect_files_for_subjects(subject_to_files, train_subjects)
    val_files = collect_files_for_subjects(subject_to_files, val_subjects)
    test_files = collect_files_for_subjects(subject_to_files, test_subjects)

    print("\nDataset split summary:")
    print(f"Total files       : {len(files)}")
    print(f"Total subjects    : {len(subject_ids)}")
    print(f"Train subjects    : {len(train_subjects)}")
    print(f"Validation subjects: {len(val_subjects)}")
    print(f"Test subjects     : {len(test_subjects)}")

    print(f"\nTrain files       : {len(train_files)}")
    print(f"Validation files  : {len(val_files)}")
    print(f"Test files        : {len(test_files)}")

    print_label_distribution("Train", train_files)
    print_label_distribution("Validation", val_files)
    print_label_distribution("Test", test_files)

    train_dataset = LazySleepSequenceDataset(
        files=train_files,
        seq_len=seq_len,
        stride=stride,
        debug=debug
    )

    val_dataset = LazySleepSequenceDataset(
        files=val_files,
        seq_len=seq_len,
        stride=stride,
        debug=debug
    )

    test_dataset = LazySleepSequenceDataset(
        files=test_files,
        seq_len=seq_len,
        stride=stride,
        debug=debug
    )

    print("\nSequence counts:")
    print(f"Train sequences     : {len(train_dataset)}")
    print(f"Validation sequences: {len(val_dataset)}")
    print(f"Test sequences      : {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    npz_dir,
    seq_len=20,
    stride=None,
    batch_size=2,
    num_workers=0,
    seed=42,
    debug=False
):
    """
    Train / validation / test DataLoader oluşturur.
    """
    train_dataset, val_dataset, test_dataset = create_datasets(
        npz_dir=npz_dir,
        seq_len=seq_len,
        stride=stride,
        seed=seed,
        debug=debug
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="Test Lazy Sleep-EDF Dataset")

    parser.add_argument("--npz_dir", type=str, default="./data/sleepedf/npz")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    print("\nDataset.py test started")
    print(f"npz_dir    : {args.npz_dir}")
    print(f"seq_len    : {args.seq_len}")
    print(f"stride     : {args.stride if args.stride is not None else args.seq_len}")
    print(f"batch_size : {args.batch_size}")
    print(f"debug      : {args.debug}")

    train_loader, val_loader, test_loader = create_dataloaders(
        npz_dir=args.npz_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        debug=args.debug
    )

    print("\nChecking one training batch...")

    X, y = next(iter(train_loader))

    print(f"Batch X shape: {X.shape}")
    print(f"Batch y shape: {y.shape}")

    expected_x_dim = 3
    expected_y_dim = 2

    assert X.dim() == expected_x_dim, f"Expected X dim {expected_x_dim}, got {X.dim()}"
    assert y.dim() == expected_y_dim, f"Expected y dim {expected_y_dim}, got {y.dim()}"

    assert X.shape[1] == args.seq_len, f"Expected seq_len {args.seq_len}, got {X.shape[1]}"
    assert y.shape[1] == args.seq_len, f"Expected seq_len {args.seq_len}, got {y.shape[1]}"

    assert X.shape[2] == 3000, f"Expected epoch length 3000, got {X.shape[2]}"

    print("\nShape check passed.")
    print("Dataset.py test completed successfully.")

class LazyFusionSequenceDataset(Dataset):
    """
    Raw EEG ve manual feature dosyalarını birlikte okur.

    Raw .npz:
        x shape: [num_epochs, 3000]
        y shape: [num_epochs]

    Manual .npz:
        x_manual shape: [num_epochs, 35]
        y shape        : [num_epochs]

    Dataset çıktısı:
        X_raw shape    : [seq_len, 3000]
        X_manual shape : [seq_len, 35]
        y shape        : [seq_len]

    DataLoader sonrası:
        X_raw shape    : [batch_size, seq_len, 3000]
        X_manual shape : [batch_size, seq_len, 35]
        y shape        : [batch_size, seq_len]
    """

    def __init__(self, raw_files, manual_dir, seq_len=20, stride=None, debug=False, max_sequences_per_file=5):
        super().__init__()

        self.raw_files = sorted(raw_files)
        self.manual_dir = manual_dir
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.debug = debug
        self.max_sequences_per_file = max_sequences_per_file

        self.index = []

        self._build_index()

    def _get_manual_file(self, raw_file):
        base_name = os.path.splitext(os.path.basename(raw_file))[0]
        manual_file = os.path.join(self.manual_dir, f"{base_name}_manual.npz")

        if not os.path.exists(manual_file):
            raise FileNotFoundError(f"Manual feature file not found: {manual_file}")

        return manual_file

    def _build_index(self):
        for raw_file in self.raw_files:
            manual_file = self._get_manual_file(raw_file)

            raw_data = np.load(raw_file)
            manual_data = np.load(manual_file)

            x_raw = raw_data["x"]
            y_raw = raw_data["y"]

            x_manual = manual_data["x_manual"]
            y_manual = manual_data["y"]

            if len(x_raw) != len(x_manual):
                raise RuntimeError(
                    f"Raw/manual epoch mismatch: {raw_file} "
                    f"raw={len(x_raw)}, manual={len(x_manual)}"
                )

            if len(y_raw) != len(y_manual):
                raise RuntimeError(
                    f"Raw/manual label mismatch: {raw_file}"
                )

            if x_manual.shape[1] != 35:
                raise RuntimeError(
                    f"Expected 35 manual features, got {x_manual.shape[1]}"
                )

            num_epochs = len(x_raw)
            max_start = num_epochs - self.seq_len

            if max_start < 0:
                continue

            starts = list(range(0, max_start + 1, self.stride))

            if self.debug:
                starts = starts[:self.max_sequences_per_file]

            subject_id = get_subject_id(raw_file)

            for start in starts:
                self.index.append({
                    "raw_file": raw_file,
                    "manual_file": manual_file,
                    "subject_id": subject_id,
                    "start": start
                })

        if len(self.index) == 0:
            raise RuntimeError("No valid fusion sequences were created.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        item = self.index[idx]

        raw_file = item["raw_file"]
        manual_file = item["manual_file"]
        start = item["start"]
        end = start + self.seq_len

        raw_data = np.load(raw_file)
        manual_data = np.load(manual_file)

        x_raw = raw_data["x"][start:end]
        y = raw_data["y"][start:end]

        x_manual = manual_data["x_manual"][start:end]

        if x_raw.shape != (self.seq_len, 3000):
            raise RuntimeError(f"Invalid x_raw shape: {x_raw.shape}")

        if x_manual.shape != (self.seq_len, 35):
            raise RuntimeError(f"Invalid x_manual shape: {x_manual.shape}")

        if y.shape[0] != self.seq_len:
            raise RuntimeError(f"Invalid y shape: {y.shape}")

        x_raw = torch.tensor(x_raw, dtype=torch.float32)
        x_manual = torch.tensor(x_manual, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x_raw, x_manual, y

if __name__ == "__main__":
    main()