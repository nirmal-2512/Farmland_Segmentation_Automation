# ============================================================
# CONFIGURATION
# ============================================================

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL = _PROJECT_ROOT / "models" / "segformer_farmland.onnx"

# ============================================================
# SERVER SETTINGS
# ============================================================

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# ============================================================
# MODEL SETTINGS
# ============================================================

ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", str(_DEFAULT_MODEL))

DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", 0.25))
DEFAULT_MIN_AREA = int(os.getenv("DEFAULT_MIN_AREA", 100))

# ============================================================
# INFERENCE SETTINGS
# ============================================================

_DEFAULT_PROVIDERS = (
    "CUDAExecutionProvider,CPUExecutionProvider"
    if os.getenv("USE_GPU", "").lower() in ("1", "true", "yes")
    else "CPUExecutionProvider"
)
INFERENCE_PROVIDERS = os.getenv("INFERENCE_PROVIDERS", _DEFAULT_PROVIDERS).split(",")

# ============================================================
# IMAGE SETTINGS
# ============================================================

MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 4096))
ALLOWED_FORMATS = ["jpg", "jpeg", "png", "bmp", "tiff"]

# ============================================================
# CORS SETTINGS
# ============================================================

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "*"
).split(",")

# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

print(f"Configuration Loaded:")
print(f"  Host: {HOST}:{PORT}")
print(f"  Model: {ONNX_MODEL_PATH}")
print(f"  Threshold: {DEFAULT_THRESHOLD}")
print(f"  Providers: {INFERENCE_PROVIDERS}")
