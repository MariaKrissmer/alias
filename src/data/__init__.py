# data/__init__.py

# Import functions from submodules
from .build_datasets import build_datasets, build_triplets
from .scrna import DatascRNAConfig
from .ncbi import DataNCBIConfig
from .triplet_generation import TripletGenerationConfig

# Optional: define what is publicly available
__all__ = [
    "build_datasets",
    "build_triplets",
    "DatascRNAConfig",
    "DataNCBIConfig",
    "TripletGenerationConfig"
]