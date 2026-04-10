import sys
from pathlib import Path

import torch

# training/pytorch on path for Palma_model + pytorch_convLSTM
_PT_DIR = Path(__file__).resolve().parents[2] / "training" / "pytorch"
sys.path.insert(0, str(_PT_DIR))

from Palma_model import MultiLayerConvLSTM

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "models" / "pytorch" / "model_pytorch.pth"
DEFAULT_ONNX = REPO_ROOT / "exports" / "onnx" / "model_pytorch.onnx"

# Load the model
model = MultiLayerConvLSTM(use_sigmoid=True)
model.load_state_dict(torch.load(DEFAULT_MODEL, map_location="cpu"))
model.eval()

# Create a sample input (batch size 1, 18 timesteps, 1 channel, 344x315)
dummy_input = torch.randn(1, 18, 1, 344, 315)

DEFAULT_ONNX.parent.mkdir(parents=True, exist_ok=True)

# Export to ONNX
torch.onnx.export(
    model,                             # Model to export
    dummy_input,                       # Example input
    str(DEFAULT_ONNX),                 # Output ONNX file path
    export_params=True,                 # Export all parameters
    opset_version=11,                   # ONNX opset version
    do_constant_folding=True,           # Optimize constant expressions
    input_names=['input'],              # Input name
    output_names=['output'],            # Output name
    dynamic_axes={
        'input': {0: 'batch_size'},     # Only the batch dimension is dynamic
        'output': {0: 'batch_size'}
    }
)

print("ONNX export completed:", DEFAULT_ONNX)
