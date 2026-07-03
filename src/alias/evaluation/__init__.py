# evaluation/__init__.py

# This makes the evaluation directory a proper Python package

# import functions from submodules
from .embedding import (
    GenEmbeddingsConfig,
    generate_embeddings,
    infer_embedding_run_timestamp,
    load_dataset_embedding_artifacts,
    load_embedding_artifact,
    load_saved_embeddings,
)
from .celltype_label_plots import umap_plots, EvaluationConfig
from .celltype_label_similarity import CellTypeSimilarityConfig, cell_type_label_similarity
from .functionality_cell_similarity import FunctionalitySimilarityConfig, functionality_similarity

try:
    from .disease_comparison import DiseaseComparisonConfig, disease_comparison
except Exception:  # Optional until the migration lands.
    DiseaseComparisonConfig = None
    disease_comparison = None

try:
    from .pseudotime import PseudotimeConfig, pseudotime
except Exception:  # Optional until the migration lands.
    PseudotimeConfig = None
    pseudotime = None

# Optional: define what is publicly available
__all__ = [
    "GenEmbeddingsConfig",
    "generate_embeddings",
    "infer_embedding_run_timestamp",
    "load_dataset_embedding_artifacts",
    "load_embedding_artifact",
    "load_saved_embeddings",
    "EvaluationConfig",
    "umap_plots",
    "CellTypeSimilarityConfig",
    "cell_type_label_similarity",
    "FunctionalitySimilarityConfig",
    "functionality_similarity",
]

if DiseaseComparisonConfig is not None and disease_comparison is not None:
    __all__.extend(
        [
            "DiseaseComparisonConfig",
            "disease_comparison",
        ]
    )

if PseudotimeConfig is not None and pseudotime is not None:
    __all__.extend(
        [
            "PseudotimeConfig",
            "pseudotime",
        ]
    )
