import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import h5py
from PIL import Image

from Palma_model import MultiLayerConvLSTM

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "pytorch" / "model_pytorch.pth"


def parse_args():
    p = argparse.ArgumentParser(description="Validate native PyTorch ConvLSTM (task RMSE/MAE).")
    p.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to model_pytorch.pth (default: models/pytorch/model_pytorch.pth)",
    )
    p.add_argument(
        "--validation-dir",
        type=Path,
        required=True,
        help="Directory of validation batch folders (see create_dataset_from_raw).",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("validation_summary.csv"),
        help="Where to write per-sample metrics.",
    )
    p.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[315, 344],
        metavar=("W", "H"),
        help="PIL resize (width, height); default matches STAC metadata 344x315 HxW stack.",
    )
    return p.parse_args()


args = parse_args()
MODEL_PATH = args.model
VALIDATION_PATH = args.validation_dir
SAVE_CSV_PATH = args.output_csv
RESIZE_TO = tuple(args.resize)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ==== Model ====
print("Loading model...")
model = MultiLayerConvLSTM(use_sigmoid=True).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded.")

# ==== Data preparation ====
def create_dataset_from_raw(directory_path, resize_to):
    """Load raw HDF5 batches (36 timesteps), resize, normalize to [0,1]."""
    resize_w, resize_h = resize_to
    batch_names = sorted(
        os.path.join(directory_path, name)
        for name in os.listdir(directory_path)
        if os.path.isdir(os.path.join(directory_path, name))
    )
    dataset = []
    for batch in batch_names:
        files = sorted(f for f in os.listdir(batch) if f.lower().endswith(".hf5"))[:36]
        if len(files) < 36:
            continue
        crn_batch = np.zeros((36, resize_h, resize_w), dtype=np.float32)
        try:
            for idx, raster in enumerate(files):
                fn = os.path.join(batch, raster)
                with h5py.File(fn, "r") as img:
                    original = np.array(img["image1"]["image_data"]).astype(np.uint8)
                    resized = Image.fromarray(original).resize((resize_w, resize_h), resample=Image.BILINEAR)
                    crn_batch[idx] = np.asarray(resized, dtype=np.float32) / 255.0
            dataset.append(crn_batch)
        except Exception as e:
            # Skip unreadable batch
            continue

    if not dataset:
        return np.empty((0, 18, 1, resize_h, resize_w)), np.empty((0, 18, 1, resize_h, resize_w))

    dataset = np.expand_dims(np.array(dataset), axis=1)         # (N, 1, 36, H, W)
    dataset = np.transpose(dataset, (0, 2, 1, 3, 4))            # (N, 36, 1, H, W)
    return dataset[:, :18], dataset[:, 18:]

def calc_rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def calc_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

print("Loading validation data...")
X_val_np, y_val_np = create_dataset_from_raw(VALIDATION_PATH, RESIZE_TO)
print("Validation data shape:", X_val_np.shape)

# ==== Validation loop ====
results = []
total_start = time.perf_counter()

for i in range(X_val_np.shape[0]):
    batch_start = time.perf_counter()

    input_seq = torch.tensor(X_val_np[i], dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, 18, 1, H, W)
    target_seq = y_val_np[i].astype(np.float32)                                        # (18, 1, H, W)

    with torch.no_grad():
        prediction = model(input_seq).squeeze(0).cpu().numpy()                         # (T, H, W, 1)
        prediction = np.transpose(prediction, (0, 3, 1, 2))                            # (T, 1, H, W)

    prediction = np.clip(prediction, 0.0, 1.0)
    rmse = calc_rmse(target_seq, prediction)
    mae = calc_mae(target_seq, prediction)

    results.append((i, rmse, mae, time.perf_counter() - batch_start))

total_time = time.perf_counter() - total_start
avg_time = (total_time / len(results)) if results else float("nan")

os.makedirs(os.path.dirname(SAVE_CSV_PATH) or ".", exist_ok=True)
with open(SAVE_CSV_PATH, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Index", "RMSE", "MAE", "Time_per_Batch_seconds"])
    writer.writerows(results)

print(f"Validated {len(results)} samples")
print(f"Average RMSE: {np.mean([r[1] for r in results]):.4f}")
print(f"Average MAE:  {np.mean([r[2] for r in results]):.4f}")
print(f"Average time per batch (s): {avg_time:.2f}")
print("Results saved to:", SAVE_CSV_PATH)
