from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from alias.evaluation.celltype_annotation.reference import (
    build_train_reference_from_dataset_dir,
    cache_metadata_matches,
    load_train_reference_metadata,
    write_json,
)


@dataclass
class CellTypistModelConfig:
    name: str
    model: Path | str
    source: Literal["local", "celltypist"] = "local"
    force_update: bool = False


@dataclass
class CellTypistAnnotationConfig:
    adata_path: Path | str
    annotation_column: str
    model_name: str
    model: Path | str
    model_source: Literal["local", "celltypist"] = "local"
    force_update: bool = False


@dataclass
class CellTypistTrainingConfig:
    dataset_dir: Path | str
    annotation_column: str
    model_path: Path | str
    model_name: str
    reference_cache_dir: Path | str | None = None
    reference_use_raw: bool = False
    reference_obs_value_map: dict[str, dict[str, str]] | None = None
    force_rebuild_reference: bool = False
    force_retrain: bool = False
    check_expression: bool = True
    C: float = 1.0
    solver: str | None = None
    max_iter: int | None = 500
    n_jobs: int | None = None
    use_SGD: bool = False
    alpha: float = 0.0001
    use_GPU: bool = False
    mini_batch: bool = False
    batch_number: int = 100
    batch_size: int = 1000
    epochs: int = 10
    balance_cell_type: bool = False
    feature_selection: bool = False
    top_genes: int = 300
    date: str = ""
    details: str = ""
    url: str = ""
    source: str = ""
    version: str = ""


def _import_celltypist_dependencies() -> tuple[Any, Any]:
    try:
        import scanpy as sc
        import celltypist
    except ImportError as exc:
        raise ImportError(
            "CellTypist annotation requires the annotation environment. "
            "Install/use the annotation dependencies before running this benchmark."
        ) from exc

    return sc, celltypist


def _celltypist_training_parameters(config: CellTypistTrainingConfig) -> dict[str, Any]:
    return {
        "check_expression": config.check_expression,
        "C": config.C,
        "solver": config.solver,
        "max_iter": config.max_iter,
        "n_jobs": config.n_jobs,
        "use_SGD": config.use_SGD,
        "alpha": config.alpha,
        "use_GPU": config.use_GPU,
        "mini_batch": config.mini_batch,
        "batch_number": config.batch_number,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "balance_cell_type": config.balance_cell_type,
        "feature_selection": config.feature_selection,
        "top_genes": config.top_genes,
        "date": config.date,
        "details": config.details,
        "url": config.url,
        "source": config.source,
        "version": config.version,
    }


def _celltypist_model_metadata(
    config: CellTypistTrainingConfig,
    n_train_cells: int,
    cell_types: list[str],
) -> dict[str, Any]:
    reference_metadata = load_train_reference_metadata(
        dataset_dir=config.dataset_dir,
        annotation_column=config.annotation_column,
        use_raw=config.reference_use_raw,
        obs_value_map=config.reference_obs_value_map,
    )
    metadata = reference_metadata.cache_fingerprint()
    metadata.update(
        {
            "model_name": config.model_name,
            "model_path": str(Path(config.model_path)),
            "reference_cache_dir": (
                str(config.reference_cache_dir)
                if config.reference_cache_dir is not None
                else str(reference_metadata.dataset_dir / "celltype_annotation" / "references")
            ),
            "n_train_cells": n_train_cells,
            "cell_types": sorted(map(str, cell_types)),
            "celltypist_train_parameters": _celltypist_training_parameters(config),
        }
    )
    return metadata


def train_celltypist_model_from_dataset_dir(
    config: CellTypistTrainingConfig,
    celltypist_module: Any | None = None,
) -> Path:
    """Train or reuse a CellTypist model from the matching train split."""
    if celltypist_module is None:
        _, celltypist = _import_celltypist_dependencies()
    else:
        celltypist = celltypist_module

    model_path = Path(config.model_path)
    metadata_path = model_path.with_suffix(".metadata.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    reference = build_train_reference_from_dataset_dir(
        dataset_dir=config.dataset_dir,
        annotation_column=config.annotation_column,
        reference_cache_dir=config.reference_cache_dir,
        force_rebuild=config.force_rebuild_reference,
        use_raw=config.reference_use_raw,
        obs_value_map=config.reference_obs_value_map,
    )
    if config.annotation_column not in reference.obs:
        raise ValueError(
            f"Annotation column {config.annotation_column!r} not found in train reference obs."
        )

    labels = reference.obs[config.annotation_column].astype(str)
    metadata = _celltypist_model_metadata(
        config=config,
        n_train_cells=reference.n_obs,
        cell_types=labels.unique().tolist(),
    )
    if (
        not config.force_retrain
        and model_path.exists()
        and cache_metadata_matches(metadata_path, metadata)
    ):
        return model_path

    train_kwargs = _celltypist_training_parameters(config)
    model = celltypist.train(
        X=reference,
        labels=labels,
        **train_kwargs,
    )
    model.write(str(model_path))
    write_json(metadata_path, metadata)
    return model_path


def resolve_celltypist_model(
    model_config: CellTypistModelConfig,
    celltypist_module: Any | None,
) -> str:
    """Resolve a CellTypist model from either a local AIFI file or model collection name."""
    if model_config.source == "local":
        model_path = Path(model_config.model)
        if not model_path.exists():
            raise FileNotFoundError(f"CellTypist model file not found: {model_path}")
        return str(model_path)

    if model_config.source == "celltypist":
        if celltypist_module is None:
            raise ValueError("celltypist_module is required for CellTypist collection models.")
        model_name = str(model_config.model)
        celltypist_module.models.download_models(
            force_update=model_config.force_update,
            model=[model_name],
        )
        return model_name

    raise ValueError(f"Unsupported CellTypist model source: {model_config.source}")


def run_celltypist_annotation(config: CellTypistAnnotationConfig) -> pd.DataFrame:
    """Run CellTypist on an AnnData file and return standardized predictions."""
    sc, celltypist = _import_celltypist_dependencies()

    adata_path = Path(config.adata_path)
    if not adata_path.exists():
        raise FileNotFoundError(f"AnnData test file not found: {adata_path}")
    resolved_model = resolve_celltypist_model(
        CellTypistModelConfig(
            name=config.model_name,
            model=config.model,
            source=config.model_source,
            force_update=config.force_update,
        ),
        celltypist_module=celltypist,
    )

    adata = sc.read_h5ad(adata_path)
    if config.annotation_column not in adata.obs:
        raise ValueError(
            f"Annotation column {config.annotation_column!r} not found in adata.obs."
        )

    predictions = celltypist.annotate(adata, model=resolved_model)
    predicted_labels = predictions.predicted_labels
    if isinstance(predicted_labels, pd.Series):
        label_series = predicted_labels
    else:
        label_column = (
            "predicted_labels"
            if "predicted_labels" in predicted_labels.columns
            else predicted_labels.columns[0]
        )
        label_series = predicted_labels[label_column]

    result = pd.DataFrame(index=adata.obs_names.astype(str))
    result.index.name = "cell_id"
    result["true_label"] = adata.obs[config.annotation_column].astype(str).to_numpy()
    result["predicted_label"] = label_series.reindex(adata.obs_names).astype(str).to_numpy()

    probability_matrix = getattr(predictions, "probability_matrix", None)
    if probability_matrix is not None:
        score_values = []
        for cell_id, predicted_label in result["predicted_label"].items():
            if cell_id in probability_matrix.index and predicted_label in probability_matrix.columns:
                score_values.append(float(probability_matrix.loc[cell_id, predicted_label]))
            else:
                score_values.append(float("nan"))
        result["score"] = score_values

    result["model_name"] = config.model_name
    return result
