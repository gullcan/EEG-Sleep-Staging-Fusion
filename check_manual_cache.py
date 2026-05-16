import glob
import os
import numpy as np

manual_dir = "./data/sleepedf/manual_features_e100"

files = sorted(glob.glob(os.path.join(manual_dir, "*_manual.npz")))

print("manual files:", len(files))

bad_files = []
total_epochs = 0

for file_path in files:
    data = np.load(file_path)

    if "x_manual" not in data or "y" not in data:
        bad_files.append((os.path.basename(file_path), "missing x_manual or y"))
        continue

    x_manual = data["x_manual"]
    y = data["y"]

    total_epochs += len(y)

    if x_manual.ndim != 2:
        bad_files.append((os.path.basename(file_path), "x_manual ndim error", x_manual.shape))
        continue

    if x_manual.shape[1] != 35:
        bad_files.append((os.path.basename(file_path), "feature dim error", x_manual.shape))
        continue

    if x_manual.shape[0] != len(y):
        bad_files.append((os.path.basename(file_path), "x/y length mismatch", x_manual.shape, y.shape))
        continue

    if np.isnan(x_manual).any():
        bad_files.append((os.path.basename(file_path), "NaN detected"))
        continue

    if np.isinf(x_manual).any():
        bad_files.append((os.path.basename(file_path), "Inf detected"))
        continue

print("total manual epochs:", total_epochs)
print("bad file count:", len(bad_files))

if bad_files:
    print("First bad files:")
    for item in bad_files[:10]:
        print(item)
else:
    print("all valid: True")