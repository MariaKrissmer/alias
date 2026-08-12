from pathlib import Path
import logging
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# evaluation/__init__.py

# This makes the evaluation directory a proper Python package

# import functions from submodules
from .embedding import (
    GenEmbeddingsConfig,
    generate_celltype_label_embedding_variant,
    generate_embeddings,
    infer_embedding_run_timestamp,
    load_dataset_embedding_artifacts,
    load_embedding_artifact,
    load_saved_embeddings,
)
from .celltype_label_plots import umap_plots, EvaluationConfig
from .embedding_subset_umap import EmbeddingSubsetUMAPConfig, embedding_subset_umap_plots
from .functionality_benchmark import (
    FunctionalityBenchmarkConfig,
    FunctionalityBenchmarkSource,
    compute_ground_truth_ranks,
    load_source_scores,
    run_functionality_benchmark,
    summarize_ground_truth_ranks,
)
try:
    from .functionality_downstream_assessment import (
        ACTIVATED_GENE_SET,
        ANTIGEN_PRESENTATION_GENE_SET,
        APOPTOTIC_SIGNALING_GENE_SET,
        CELL_PROLIFERATION_GENE_SET,
        CELLULAR_STRESS_RESPONSE_GENE_SET,
        CYTOKINE_PRODUCTION_GENE_SET,
        CYTOTOXIC_GENE_SET,
        EXHAUSTED_GENE_SET,
        FunctionalityDownstreamConfig,
        HELPER_GENE_SET,
        INFLAMMATORY_GENE_SET,
        INHIBITORY_GENE_SET,
        IMMUNOSUPPRESSIVE_GENE_SET,
        REGULATORY_GENE_SET,
        TISSUE_MIGRATION_GENE_SET,
        TYPE_I_INTERFERON_RESPONSE_GENE_SET,
        compute_cutoff,
        run_functionality_downstream_assessment,
    )
except Exception:  # Optional in lightweight contexts without scanpy/anndata.
    ACTIVATED_GENE_SET = None
    ANTIGEN_PRESENTATION_GENE_SET = None
    APOPTOTIC_SIGNALING_GENE_SET = None
    CELL_PROLIFERATION_GENE_SET = None
    CELLULAR_STRESS_RESPONSE_GENE_SET = None
    CYTOKINE_PRODUCTION_GENE_SET = None
    CYTOTOXIC_GENE_SET = None
    EXHAUSTED_GENE_SET = None
    HELPER_GENE_SET = None
    INFLAMMATORY_GENE_SET = None
    INHIBITORY_GENE_SET = None
    IMMUNOSUPPRESSIVE_GENE_SET = None
    REGULATORY_GENE_SET = None
    TISSUE_MIGRATION_GENE_SET = None
    TYPE_I_INTERFERON_RESPONSE_GENE_SET = None
    FunctionalityDownstreamConfig = None
    compute_cutoff = None
    run_functionality_downstream_assessment = None

try:
    from .celltype_label_similarity import CellTypeSimilarityConfig, cell_type_label_similarity
except Exception:  # Optional in lightweight benchmark/import contexts.
    CellTypeSimilarityConfig = None
    cell_type_label_similarity = None

try:
    from .functionality_cell_similarity import FunctionalitySimilarityConfig, functionality_similarity
except Exception:  # Optional in lightweight benchmark/import contexts.
    FunctionalitySimilarityConfig = None
    functionality_similarity = None

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

try:
    from .lamanno_timepoint_similarity import (
        LaMannoTimepointSimilarityConfig,
        lamanno_timepoint_similarity,
    )
except Exception:  # Optional until the migration lands.
    LaMannoTimepointSimilarityConfig = None
    lamanno_timepoint_similarity = None

try:
    from .heldout_timepoint_umap import heldout_timepoint_umap_plots
except Exception:  # Optional until the migration lands.
    heldout_timepoint_umap_plots = None

try:
    from .heldout_timepoint_quantification import (
        HeldoutTimepointNeighborConfig,
        heldout_timepoint_neighbor_enrichment,
    )
except Exception:  # Optional until the migration lands.
    HeldoutTimepointNeighborConfig = None
    heldout_timepoint_neighbor_enrichment = None

# Optional: define what is publicly available
__all__ = [
    "GenEmbeddingsConfig",
    "generate_celltype_label_embedding_variant",
    "generate_embeddings",
    "infer_embedding_run_timestamp",
    "load_dataset_embedding_artifacts",
    "load_embedding_artifact",
    "load_saved_embeddings",
    "EvaluationConfig",
    "umap_plots",
    "EmbeddingSubsetUMAPConfig",
    "embedding_subset_umap_plots",
    "FunctionalityBenchmarkConfig",
    "FunctionalityBenchmarkSource",
    "compute_ground_truth_ranks",
    "load_source_scores",
    "run_functionality_benchmark",
    "summarize_ground_truth_ranks",
]

if FunctionalityDownstreamConfig is not None and run_functionality_downstream_assessment is not None:
    __all__.extend(
        [
            "CYTOTOXIC_GENE_SET",
            "INHIBITORY_GENE_SET",
            "HELPER_GENE_SET",
            "EXHAUSTED_GENE_SET",
            "ACTIVATED_GENE_SET",
            "INFLAMMATORY_GENE_SET",
            "IMMUNOSUPPRESSIVE_GENE_SET",
            "REGULATORY_GENE_SET",
            "TYPE_I_INTERFERON_RESPONSE_GENE_SET",
            "CELL_PROLIFERATION_GENE_SET",
            "CYTOKINE_PRODUCTION_GENE_SET",
            "TISSUE_MIGRATION_GENE_SET",
            "ANTIGEN_PRESENTATION_GENE_SET",
            "CELLULAR_STRESS_RESPONSE_GENE_SET",
            "APOPTOTIC_SIGNALING_GENE_SET",
            "FunctionalityDownstreamConfig",
            "compute_cutoff",
            "run_functionality_downstream_assessment",
        ]
    )

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

if LaMannoTimepointSimilarityConfig is not None and lamanno_timepoint_similarity is not None:
    __all__.extend(
        [
            "LaMannoTimepointSimilarityConfig",
            "lamanno_timepoint_similarity",
        ]
    )

if HeldoutTimepointNeighborConfig is not None and heldout_timepoint_neighbor_enrichment is not None:
    __all__.extend(
        [
            "HeldoutTimepointNeighborConfig",
            "heldout_timepoint_neighbor_enrichment",
        ]
    )

if heldout_timepoint_umap_plots is not None:
    __all__.extend(["heldout_timepoint_umap_plots"])

if CellTypeSimilarityConfig is not None and cell_type_label_similarity is not None:
    __all__.extend(
        [
            "CellTypeSimilarityConfig",
            "cell_type_label_similarity",
        ]
    )

if FunctionalitySimilarityConfig is not None and functionality_similarity is not None:
    __all__.extend(
        [
            "FunctionalitySimilarityConfig",
            "functionality_similarity",
        ]
    )
