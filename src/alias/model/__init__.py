# model/__init__.py

# Import functions from submodules
from .training import train_model, TrainingSTConfig

# Optional: define what is publicly available
__all__ = [
    "train_model",
    "TrainingSTConfig"
]