import argparse
import glob
import os
import math

import numpy as np
import pywt
from scipy.signal import butter, filtfilt, welch, hilbert


EPS = 1e-8


def compact_signal_for_entropy(signal, max_len=300):
    """
    Entropy hesaplarını hızlandırmak için sinyali sabit uzunluğa indirir.

    Input:
        signal shape: [N]

    Output:
        compact signal shape: [max_len] veya daha kısa
    """
    signal = np.asarray(signal)

    if len(signal) <= max_len:
        return signal

    idx = np.linspace(0, len(signal) - 1, max_len).astype(int)
    return signal[idx]


def bandpass_filter(signal, fs=100, lowcut=0.3, highcut=45.0, order=4):
    """
    EEG sinyaline bandpass filter uygular.

    Input:
        signal shape: [3000]

    Output:
        filtered_signal shape: [3000]
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype="band")
    filtered = filtfilt(b, a, signal)

    return filtered


def wavelet_decompose(signal, wavelet="db4", level=4):
    """
    4-level wavelet decomposition yapar.

    Makaledeki mantığa göre:
        cA4 -> delta
        cD4 -> theta
        cD3 -> alpha
        cD2 -> beta

    Not:
        cD1 yüksek frekans bileşenidir ve bu 35 feature setine dahil edilmiyor.
    """
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)

    cA4, cD4, cD3, cD2, cD1 = coeffs

    delta = pywt.upcoef("a", cA4, wavelet, level=4, take=len(signal))
    theta = pywt.upcoef("d", cD4, wavelet, level=4, take=len(signal))
    alpha = pywt.upcoef("d", cD3, wavelet, level=3, take=len(signal))
    beta = pywt.upcoef("d", cD2, wavelet, level=2, take=len(signal))

    return {
        "original": signal,
        "delta": delta,
        "theta": theta,
        "alpha": alpha,
        "beta": beta
    }


def energy(signal):
    """
    Ortalama enerji.
    """
    signal = np.asarray(signal)
    return np.mean(signal ** 2)


def _phi(signal, m, r):
    """
    Approximate entropy için yardımcı fonksiyon.
    """
    signal = np.asarray(signal)
    n = len(signal)

    if n <= m + 1:
        return 0.0

    patterns = np.array([signal[i:i + m] for i in range(n - m + 1)])

    count = []
    for pattern in patterns:
        dist = np.max(np.abs(patterns - pattern), axis=1)
        count.append(np.mean(dist <= r))

    count = np.asarray(count)
    return np.mean(np.log(count + EPS))


def approximate_entropy(signal, m=2, r_ratio=0.2):
    """
    Approximate Entropy.

    r genellikle 0.2 * std olarak seçilir.
    """
    signal = np.asarray(signal)
    std = np.std(signal)

    if std < EPS:
        return 0.0

    r = r_ratio * std

    return _phi(signal, m, r) - _phi(signal, m + 1, r)


def fuzzy_entropy(signal, m=2, r_ratio=0.2, n_exp=2):
    """
    Fuzzy Entropy.

    Not:
        Makalede tüm entropy parametreleri açık verilmediği için
        standart parametreler kullanılmıştır.
    """
    signal = np.asarray(signal)
    std = np.std(signal)

    if std < EPS:
        return 0.0

    r = r_ratio * std

    def _fuzzy_phi(m):
        patterns = np.array([signal[i:i + m] for i in range(len(signal) - m + 1)])
        patterns = patterns - np.mean(patterns, axis=1, keepdims=True)

        fuzzy_values = []

        for pattern in patterns:
            dist = np.max(np.abs(patterns - pattern), axis=1)
            similarity = np.exp(-(dist ** n_exp) / (r + EPS))
            fuzzy_values.append(np.mean(similarity))

        return np.mean(fuzzy_values)

    return np.log((_fuzzy_phi(m) + EPS) / (_fuzzy_phi(m + 1) + EPS))


def coarse_grain(signal, scale):
    """
    Multiscale entropy için coarse-graining işlemi.
    """
    signal = np.asarray(signal)
    n = len(signal) // scale

    if n == 0:
        return signal

    return np.mean(signal[:n * scale].reshape(n, scale), axis=1)


def multiscale_entropy(signal, scales=(1, 2, 3), m=2, r_ratio=0.2):
    """
    Basit multiscale entropy yaklaşımı.
    Her scale için approximate entropy hesaplanır ve ortalaması alınır.
    """
    values = []

    for scale in scales:
        cg_signal = coarse_grain(signal, scale)
        values.append(approximate_entropy(cg_signal, m=m, r_ratio=r_ratio))

    return np.mean(values)


def permutation_entropy(signal, order=3, delay=1):
    """
    Permutation Entropy.
    """
    signal = np.asarray(signal)
    n = len(signal)

    if n < order * delay:
        return 0.0

    patterns = []

    for i in range(n - delay * (order - 1)):
        window = signal[i:i + delay * order:delay]
        patterns.append(tuple(np.argsort(window)))

    _, counts = np.unique(patterns, axis=0, return_counts=True)
    probs = counts / np.sum(counts)

    pe = -np.sum(probs * np.log(probs + EPS))

    # normalize
    pe = pe / np.log(math.factorial(order))

    return pe


def power_spectral_entropy(signal, fs=100):
    """
    Welch PSD üzerinden spectral entropy hesaplar.
    """
    freqs, psd = welch(signal, fs=fs, nperseg=256)

    psd = psd + EPS
    psd_norm = psd / np.sum(psd)

    entropy = -np.sum(psd_norm * np.log(psd_norm))
    entropy = entropy / np.log(len(psd_norm))

    return entropy


def envelope_entropy(signal):
    """
    Hilbert transform ile envelope entropy hesaplar.
    """
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)

    envelope = envelope + EPS
    probs = envelope / np.sum(envelope)

    entropy = -np.sum(probs * np.log(probs))
    entropy = entropy / np.log(len(probs))

    return entropy


def extract_7_features(signal, fs=100, entropy_max_len=300):
    """
    Bir sinyalden 7 manual feature çıkarır.

    Energy, spectral entropy ve envelope entropy tam sinyalden hesaplanır.
    Hesaplama maliyeti yüksek entropy özellikleri sabit uzunluklu kompakt sinyalden hesaplanır.

    Output shape:
        [7]
    """
    compact_signal = compact_signal_for_entropy(signal, max_len=entropy_max_len)

    return np.array([
        energy(signal),
        approximate_entropy(compact_signal),
        fuzzy_entropy(compact_signal),
        multiscale_entropy(compact_signal),
        permutation_entropy(compact_signal),
        power_spectral_entropy(signal, fs=fs),
        envelope_entropy(signal)
    ], dtype=np.float32)


def extract_manual_features(epoch, fs=100, entropy_max_len=300):
    """
    Tek bir EEG epoch'u için 35-dimensional manual feature vector çıkarır.

    Input:
        epoch shape: [3000]

    Output:
        features shape: [35]
    """
    epoch = np.asarray(epoch, dtype=np.float32)

    if epoch.ndim != 1:
        raise ValueError(f"Expected 1D epoch, got shape {epoch.shape}")

    filtered_epoch = bandpass_filter(epoch, fs=fs)

    rhythm_signals = wavelet_decompose(filtered_epoch)

    feature_list = []

    for signal_name in ["original", "delta", "theta", "alpha", "beta"]:
        signal_features = extract_7_features(rhythm_signals[signal_name], fs=fs, entropy_max_len=entropy_max_len)
        feature_list.append(signal_features)

    features = np.concatenate(feature_list, axis=0)

    if features.shape[0] != 35:
        raise RuntimeError(f"Expected 35 features, got {features.shape[0]}")

    if np.isnan(features).any():
        raise RuntimeError("NaN detected in manual features.")

    if np.isinf(features).any():
        raise RuntimeError("Inf detected in manual features.")

    return features


def process_npz_file(input_file, output_file, fs=100, entropy_max_len=300, max_epochs=None):
    """
    Tek bir .npz dosyasındaki tüm epoch'lar için 35-dimensional manual feature çıkarır.

    Input .npz:
        x shape: [num_epochs, 3000]
        y shape: [num_epochs]

    Output .npz:
        x_manual shape: [num_epochs, 35]
        y shape       : [num_epochs]
    """
    data = np.load(input_file)
    x = data["x"]
    y = data["y"]

    if max_epochs is not None:
        x = x[:max_epochs]
        y = y[:max_epochs]

    manual_features = []

    for epoch_idx, epoch in enumerate(x):
        features = extract_manual_features(
            epoch,
            fs=fs,
            entropy_max_len=entropy_max_len
        )

        manual_features.append(features)

        if (epoch_idx + 1) % 50 == 0:
            print(f"  Processed {epoch_idx + 1}/{len(x)} epochs")

    x_manual = np.vstack(manual_features).astype(np.float32)

    if x_manual.shape[0] != len(y):
        raise RuntimeError("x_manual and y length mismatch.")

    if x_manual.shape[1] != 35:
        raise RuntimeError(f"Expected 35 manual features, got {x_manual.shape[1]}")

    np.savez(
    output_file,
    x_manual=x_manual,
    y=y,
    fs=fs,
    entropy_max_len=entropy_max_len,
    feature_dim=35
)

    print(f"Saved: {output_file}")
    print(f"x_manual shape: {x_manual.shape}")
    print(f"y shape       : {y.shape}")


def is_valid_manual_cache(input_file, output_file, expected_entropy_max_len=None):
    """
    Cache dosyası gerçekten tam mı kontrol eder.

    Input .npz:
        x shape: [num_epochs, 3000]

    Output manual .npz:
        x_manual shape: [num_epochs, 35]
        y shape       : [num_epochs]
    """
    if not os.path.exists(output_file):
        return False

    try:
        input_data = np.load(input_file)
        output_data = np.load(output_file)

        source_epochs = len(input_data["x"])

        if "x_manual" not in output_data or "y" not in output_data:
            return False

        x_manual = output_data["x_manual"]
        y = output_data["y"]

        if x_manual.shape[0] != source_epochs:
            return False

        if x_manual.shape[1] != 35:
            return False

        if len(y) != source_epochs:
            return False

        if expected_entropy_max_len is not None:
            if "entropy_max_len" not in output_data:
                return False

            cached_entropy_max_len = int(output_data["entropy_max_len"])

            if cached_entropy_max_len != expected_entropy_max_len:
                return False

        if np.isnan(x_manual).any() or np.isinf(x_manual).any():
            return False

        return True

    except Exception:
        return False

def build_manual_feature_cache(
    npz_dir,
    output_dir,
    fs=100,
    entropy_max_len=300,
    max_files=None,
    max_epochs=None
):
    """
    Tüm .npz dosyaları için manual feature cache oluşturur.
    """
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(npz_dir, "*.npz")))

    if max_files is not None:
        files = files[:max_files]

    if len(files) == 0:
        raise FileNotFoundError(f"No npz files found in {npz_dir}")

    print(f"Total files to process: {len(files)}")
    print(f"Output directory: {output_dir}")

    for file_idx, input_file in enumerate(files):
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}_manual.npz")

        print("\n" + "=" * 60)
        print(f"[{file_idx + 1}/{len(files)}] Processing: {input_file}")

        if is_valid_manual_cache(
            input_file,
            output_file,
            expected_entropy_max_len=entropy_max_len
        ):
            print(f"Valid cache exists, skipping: {output_file}")
            continue

        if os.path.exists(output_file):
            print(f"Incomplete or invalid cache found. Recomputing: {output_file}")
            os.remove(output_file)

        process_npz_file(
            input_file=input_file,
            output_file=output_file,
            fs=fs,
            entropy_max_len=entropy_max_len,
            max_epochs=max_epochs
        )

def main():
    parser = argparse.ArgumentParser(description="35-dimensional manual EEG feature extraction")
    parser.add_argument("--npz_dir", type=str, default="./data/sleepedf/npz")
    parser.add_argument("--output_dir", type=str, default="./data/sleepedf/manual_features")
    parser.add_argument("--fs", type=int, default=100)
    parser.add_argument("--entropy_max_len", type=int, default=300)
    parser.add_argument("--build_cache", action="store_true")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    args = parser.parse_args()

    if args.build_cache:
        build_manual_feature_cache(
            npz_dir=args.npz_dir,
            output_dir=args.output_dir,
            fs=args.fs,
            entropy_max_len=args.entropy_max_len,
            max_files=args.max_files,
            max_epochs=args.max_epochs
        )
        return

    files = sorted(glob.glob(os.path.join(args.npz_dir, "*.npz")))

    if len(files) == 0:
        raise FileNotFoundError(f"No npz files found in {args.npz_dir}")

    first_file = files[0]
    print(f"Using file: {first_file}")

    data = np.load(first_file)
    x = data["x"]
    y = data["y"]

    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")

    epoch = x[0]
    label = y[0]

    print(f"Single epoch shape: {epoch.shape}")
    print(f"Single epoch label: {label}")

    features = extract_manual_features(
        epoch,
        fs=args.fs,
        entropy_max_len=args.entropy_max_len
    )

    print(f"Manual feature shape: {features.shape}")
    print("Manual features:")
    print(np.round(features, 4))

    print("\nFeature extraction test completed successfully.")

if __name__ == "__main__":
    main()