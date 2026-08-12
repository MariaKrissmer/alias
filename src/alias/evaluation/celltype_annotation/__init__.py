"""Cell-type annotation benchmark helpers."""

from .benchmark import (
    REQUIRED_EVALUATION_FILES,
    REQUIRED_PREDICTION_FILE,
    get_annotation_cache_status,
    make_similarity_top_label_predictions,
    write_annotation_evaluation,
)
from .celltypist import (
    CellTypistAnnotationConfig,
    CellTypistModelConfig,
    CellTypistTrainingConfig,
    resolve_celltypist_model,
    run_celltypist_annotation,
    train_celltypist_model_from_dataset_dir,
)
from .singler import (
    SingleRAnnotationConfig,
    build_singler_reference_from_dataset_dir,
    run_singler_annotation,
)
from .sctype import (
    DEFAULT_HIAI_AIFI_L2_MARKERS,
    ScTypeAnnotationConfig,
    load_sctype_marker_map,
    run_sctype_annotation,
)

__all__ = [
    "REQUIRED_EVALUATION_FILES",
    "REQUIRED_PREDICTION_FILE",
    "get_annotation_cache_status",
    "make_similarity_top_label_predictions",
    "write_annotation_evaluation",
    "CellTypistAnnotationConfig",
    "CellTypistModelConfig",
    "CellTypistTrainingConfig",
    "resolve_celltypist_model",
    "run_celltypist_annotation",
    "train_celltypist_model_from_dataset_dir",
    "SingleRAnnotationConfig",
    "build_singler_reference_from_dataset_dir",
    "run_singler_annotation",
    "DEFAULT_HIAI_AIFI_L2_MARKERS",
    "ScTypeAnnotationConfig",
    "load_sctype_marker_map",
    "run_sctype_annotation",
]
