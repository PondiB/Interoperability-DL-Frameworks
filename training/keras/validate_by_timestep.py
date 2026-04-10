import argparse
import csv
import os
from pathlib import Path

import numpy as np
from PIL import Image
import h5py

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "models" / "keras_pt" / "model_keras3_pt.keras"


def parse_args():
    p = argparse.ArgumentParser(description="Per-timestep task RMSE/MAE (Keras PT).")
    p.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to model_keras3_pt.keras",
    )
    p.add_argument(
        "--validation-dir",
        type=Path,
        required=True,
        help="Directory of validation batch folders.",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("error_by_timestep.csv"),
        help="Output CSV path.",
    )
    p.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[315, 344],
        metavar=("W", "H"),
    )
    return p.parse_args()


args = parse_args()
MODEL_PATH = args.model
VALIDATION_PATH = args.validation_dir
SAVE_CSV_PATH = args.output_csv
RESIZE_TO = tuple(args.resize)

# ==== Load model ====
print("Keras backend:", keras.backend.backend())
print("Loading model...")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# ==== Load validation data ====
def load_validation_data(directory_path, resize_to):
    resize_width, resize_height = resize_to
    X, y = [], []
    for folder in sorted(os.listdir(directory_path)):
        folder_path = os.path.join(directory_path, folder)
        if not os.path.isdir(folder_path):
            continue
        files = sorted(
            f
            for f in os.listdir(folder_path)
            if f.lower().endswith((".hf5", ".h5", ".hdf5"))
        )[:36]
        if len(files) < 36:
            continue
        sequence = np.zeros((36, resize_height, resize_width), dtype=np.float32)
        for i, file in enumerate(files):
            with h5py.File(os.path.join(folder_path, file), "r") as hf:
                data = np.array(hf["image1"]["image_data"]).astype(np.uint8)
                resized = Image.fromarray(data).resize((resize_width, resize_height), resample=Image.BILINEAR)
                sequence[i] = np.asarray(resized, dtype=np.float32) / 255.0
        X.append(sequence[:18])
        y.append(sequence[18:])
    X = np.expand_dims(np.array(X), axis=-1)
    y = np.expand_dims(np.array(y), axis=-1)
    return X, y

print("Loading validation data...")
X_val, y_val = load_validation_data(VALIDATION_PATH, RESIZE_TO)
print("Validation data shape:", X_val.shape)

# ==== Calculate errors per timestep ====
rmse_list = []
mae_list = []

preds = model.predict(X_val, batch_size=1, verbose=0)  # shape: (N, 18, H, W, 1)
for t in range(18):
    pred_t = preds[:, t, :, :, 0]
    true_t = y_val[:, t, :, :, 0]
    rmse = np.sqrt(np.mean((pred_t - true_t) ** 2))
    mae = np.mean(np.abs(pred_t - true_t))
    rmse_list.append(rmse)
    mae_list.append(mae)

# ==== Save CSV ====
SAVE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(SAVE_CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestep", "RMSE", "MAE"])
    for i in range(18):
        writer.writerow([i + 1, rmse_list[i], mae_list[i]])

print("Per-timestep errors saved to:", SAVE_CSV_PATH)
