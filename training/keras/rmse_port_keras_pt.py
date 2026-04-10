"""
RMSE_port and MAE_port between Keras 3 (PyTorch backend) and ONNX Runtime
for the same ConvLSTM weights, per the paper (output-level portability).

  RMSE_port = sqrt( mean_{i,k} (y_native_ik - y_onnx_ik)^2 )
  MAE_port  = mean_{i,k} |y_native_ik - y_onnx_ik|

Set KERAS_BACKEND=torch before importing keras (done below).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

# Must run before keras import
os.environ.setdefault("KERAS_BACKEND", "torch")
# macOS MPS lacks linalg_qr used when rebuilding layers from a .keras config; allow CPU fallback.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch

torch.set_default_device("cpu")

import numpy as np
import keras
import onnxruntime as ort
from PIL import Image
import h5py

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KERAS = REPO_ROOT / "models" / "keras_pt" / "model_keras3_pt.keras"
DEFAULT_ONNX = REPO_ROOT / "models" / "keras_pt" / "model_keras3_pt.onnx"


def load_validation_inputs(
    directory_path: Path,
    resize_to: tuple[int, int],
) -> np.ndarray:
    """Load input sequences only: (N, 18, H, W, 1) float32, channels-last."""
    resize_width, resize_height = resize_to
    xs: list[np.ndarray] = []
    for folder in sorted(directory_path.iterdir()):
        if not folder.is_dir():
            continue
        files = sorted(
            f
            for f in folder.iterdir()
            if f.suffix.lower() in (".hf5", ".h5", ".hdf5")
        )[:36]
        if len(files) < 36:
            continue
        sequence = np.zeros((36, resize_height, resize_width), dtype=np.float32)
        for i, file in enumerate(files):
            with h5py.File(file, "r") as hf:
                data = np.array(hf["image1"]["image_data"]).astype(np.uint8)
                resized = Image.fromarray(data).resize(
                    (resize_width, resize_height), resample=Image.BILINEAR
                )
                sequence[i] = np.asarray(resized, dtype=np.float32) / 255.0
        xs.append(sequence[:18])
    if not xs:
        return np.empty((0, 18, resize_height, resize_width, 1), dtype=np.float32)
    x = np.expand_dims(np.array(xs, dtype=np.float32), axis=-1)
    return x


def export_onnx_if_needed(
    model: keras.Model,
    out_path: Path,
    sample: np.ndarray,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample = np.ascontiguousarray(sample.astype(np.float32))
    model(sample[:1])  # warm-up trace
    model.export(str(out_path), format="onnx", sample_input=sample[:1])
    print("Wrote ONNX:", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RMSE_port / MAE_port: Keras PT native vs ONNX Runtime."
    )
    p.add_argument(
        "--keras-model",
        type=Path,
        default=DEFAULT_KERAS,
        help="Path to model_keras3_pt.keras",
    )
    p.add_argument(
        "--onnx-model",
        type=Path,
        default=DEFAULT_ONNX,
        help="Path to ONNX file (exported from the same Keras checkpoint)",
    )
    p.add_argument(
        "--validation-dir",
        type=Path,
        default=None,
        help="Validation batch folders (same layout as validate_by_timestep.py)",
    )
    p.add_argument(
        "--synthetic",
        type=int,
        default=None,
        metavar="N",
        help="If set, use N random sequences [0,1] instead of --validation-dir (smoke test)",
    )
    p.add_argument(
        "--resize",
        type=int,
        nargs=2,
        default=[315, 344],
        metavar=("W", "H"),
        help="PIL resize (width, height)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for native and ONNX inference",
    )
    p.add_argument(
        "--export-onnx",
        action="store_true",
        help="If set, export ONNX from the Keras model when --onnx-model is missing",
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("path.csv"),
        help="One-row RMSE_port / MAE_port summary (default: path.csv in cwd)",
    )
    p.add_argument(
        "--no-output-csv",
        action="store_true",
        help="Skip writing the summary CSV",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of sequences (for quick tests)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    resize = tuple(args.resize)

    if args.synthetic is not None and args.synthetic < 1:
        print("--synthetic N requires N >= 1", file=sys.stderr)
        sys.exit(2)
    if args.validation_dir is None and args.synthetic is None:
        print("Provide --validation-dir or --synthetic N", file=sys.stderr)
        sys.exit(2)
    if args.validation_dir is not None and args.synthetic is not None:
        print("Use only one of --validation-dir or --synthetic", file=sys.stderr)
        sys.exit(2)

    if not args.keras_model.is_file():
        print("Keras model not found:", args.keras_model, file=sys.stderr)
        sys.exit(1)

    print("Keras backend:", keras.backend.backend())
    print("Loading Keras model:", args.keras_model)
    model = keras.models.load_model(args.keras_model)

    if args.synthetic is not None:
        h, w = resize[1], resize[0]  # height, width from (W,H)
        X = np.random.default_rng(0).random(
            (args.synthetic, 18, h, w, 1), dtype=np.float32
        )
    else:
        assert args.validation_dir is not None
        X = load_validation_inputs(args.validation_dir, resize)
        if X.size == 0:
            print(
                "No validation sequences found under",
                args.validation_dir,
                file=sys.stderr,
            )
            sys.exit(2)
    if args.max_samples is not None:
        X = X[: args.max_samples]

    if not args.onnx_model.is_file():
        if args.export_onnx:
            export_onnx_if_needed(model, args.onnx_model, X)
        else:
            print(
                "ONNX not found. Pass --export-onnx or place a file at",
                args.onnx_model,
                file=sys.stderr,
            )
            sys.exit(3)

    sess_opts = ort.SessionOptions()
    sess_opts.log_severity_level = 3
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        session = ort.InferenceSession(
            str(args.onnx_model), sess_options=sess_opts, providers=providers
        )
    except Exception:
        session = ort.InferenceSession(
            str(args.onnx_model), sess_options=sess_opts, providers=["CPUExecutionProvider"]
        )
    in_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    print("ONNX inputs:", [(i.name, i.shape, i.type) for i in session.get_inputs()])
    print("ONNX outputs:", [(o.name, o.shape, o.type) for o in session.get_outputs()])
    print("Active ORT providers:", session.get_providers())

    sse = 0.0
    mae_sum = 0.0
    n_el = 0
    bs = max(1, args.batch_size)
    n = X.shape[0]

    for start in range(0, n, bs):
        xb = np.ascontiguousarray(X[start : start + bs].astype(np.float32))
        y_native = model.predict(xb, verbose=0)
        y_native = np.asarray(y_native, dtype=np.float32)
        y_onnx = session.run([out_name], {in_name: xb})[0]
        y_onnx = np.asarray(y_onnx, dtype=np.float32)
        if y_native.shape != y_onnx.shape:
            print(
                "Shape mismatch: native",
                y_native.shape,
                "onnx",
                y_onnx.shape,
                file=sys.stderr,
            )
            sys.exit(4)
        diff = y_native - y_onnx
        sse += float(np.sum(diff * diff))
        mae_sum += float(np.sum(np.abs(diff)))
        n_el += diff.size

    rmse_port = (sse / n_el) ** 0.5
    mae_port = mae_sum / n_el
    print(f"Samples: {n}  Elements compared: {n_el}")
    print(f"RMSE_port: {rmse_port:.8e}")
    print(f"MAE_port:  {mae_port:.8e}")

    if not args.no_output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "n_samples",
                    "n_elements",
                    "RMSE_port",
                    "MAE_port",
                    "keras_model",
                    "onnx_model",
                ]
            )
            w.writerow(
                [
                    n,
                    n_el,
                    rmse_port,
                    mae_port,
                    str(args.keras_model),
                    str(args.onnx_model),
                ]
            )
        print("Wrote", args.output_csv)


if __name__ == "__main__":
    main()
