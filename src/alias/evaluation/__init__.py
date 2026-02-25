# evaluation/__init__.py

# This makes the evaluation directory a proper Python package

# import functions from submodules
from .embedding import GenEmbeddingsConfig, generate_embeddings
from .celltype_label_plots import umap_plots, EvaluationConfig
from .celltype_label_similarity import CellTypeSimilarityConfig, cell_type_label_similarity
from .functionality_cell_similarity import FunctionalitySimilarityConfig, functionality_similarity

# Optional: define what is publicly available
__all__ = [
    "GenEmbeddingsConfig",
    "generate_embeddings",
    "EvaluationConfig",
    "umap_plots",
    "CellTypeSimilarityConfig",
    "cell_type_label_similarity",
    "FunctionalitySimilarityConfig",
    "functionality_similarity"
]