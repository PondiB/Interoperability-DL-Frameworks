"""Export Keras 3 (PyTorch backend) ConvLSTM to ONNX. Run with: uv run python exports/export_scripts/keras_to_onnx.py"""

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KERAS = REPO_ROOT / "models" / "keras_pt" / "model_keras3_pt.keras"
DEFAULT_ONNX = REPO_ROOT / "models" / "keras_pt" / "model_keras3_pt.onnx"

# Load model
if not DEFAULT_KERAS.is_file():
    print("Keras model not found:", DEFAULT_KERAS, file=sys.stderr)
    sys.exit(1)

model = keras.models.load_model(DEFAULT_KERAS)

# Create sample input with correct shape (including batch dimension)
dummy_input = np.random.rand(1, 18, 344, 315, 1).astype(np.float32)

# Run a forward pass with the sample input
model(dummy_input)

DEFAULT_ONNX.parent.mkdir(parents=True, exist_ok=True)

# Export model to ONNX format
model.export(
    str(DEFAULT_ONNX),
    format="onnx",
    sample_input=dummy_input,
)

print("ONNX export completed:", DEFAULT_ONNX)
