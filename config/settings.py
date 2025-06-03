"""
Settings for the face liveness detection system.
"""

import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_CHECKPOINT_PATH = MODELS_DIR / "ckpt_iter.pth.tar"

# Application settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "False").lower() in ("true", "1", "t")
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")

# Liveness detection settings
LIVENESS_THRESHOLD = float(os.getenv("LIVENESS_THRESHOLD", "0.5"))
IMAGE_SIZE = 224
NUM_CLASSES = 2 