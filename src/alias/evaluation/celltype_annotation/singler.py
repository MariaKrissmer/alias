from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from alias.evaluation.celltype_annotation.reference import (
    build_train_reference_from_dataset_dir,
)


@dataclass
class SingleRAnnotationConfig:
    adata_path: Path | str
    annotation_column: str
    model_name: str
    reference_adata_path: Path | str | None = None
    dataset_dir: Path | str | None = None
    reference_cache_dir: Path | str | None = None
    reference_use_raw: bool = False
    reference_obs_value_map: dict[str, dict[str, str]] | None = None
    force_rebuild_reference: bool = False
    max_reference_cells_per_label: int | None = None
    reference_sample_seed: int = 42
    num_threads: int = 1
    train_args: dict[str, Any] | None = None
    classify_args: dict[str, Any] | None = None


def _import_scanpy_dependency() -> Any:
    try:
        import scanpy as sc
    except ImportError as exc:
        raise ImportError(
            "SingleR annotation requires scanpy in the annotation environment."
        ) from exc

    return sc


def _import_singler_dependency() -> Any:
    try:
        import singler
    except ImportError as exc:
        raise ImportError(
            "SingleR annotation requires the annotation environment. "
            "Install/use the annotation dependencies, including singler, before running this benchmark."
        ) from exc

    return singler


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _scrna_config_from_metadata(config_dict: dict[str, Any]):
    from alias.data.scrna import DatascRNAConfig

    allowed = {field.name for field in fields(DatascRNAConfig)}
    clean = {key: value for key, value in config_dict.items() if key in allowed}
    return DatascRNAConfig(**clean)


def _cache_metadata_matches(
    metadata_path: Path,
    *,
    source_path: Path,
    split_path: Path,
    generation_path: Path,
    annotation_column: str,
    train_indices: list[str],
) -> bool:
    if not metadata_path.exists():
        return False

    try:
        metadata = _read_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return False

    expected = {
        "source_path": str(source_path),
        "split_indices_path": str(split_path),
        "generation_metadata_path": str(generation_path),
        "annotation_column": annotation_column,
        "n_train_cells": len(train_indices),
        "source_mtime": source_path.stat().st_mtime if source_path.exists() else None,
        "split_indices_mtime": split_path.stat().st_mtime if split_path.exists() else None,
        "generation_metadata_mtime": generation_path.stat().st_mtime if generation_path.exists() else None,
    }
    return all(metadata.get(key) == value for key, value in expected.items())


def _write_cache_metadata(
    metadata_path: Path,
    *,
    source_path: Path,
    split_path: Path,
    generation_path: Path,
    annotation_column: str,
    train_indices: list[str],
) -> None:
    metadata = {
        "source_path": str(source_path),
        "split_indices_path": str(split_path),
        "generation_metadata_path": str(generation_path),
        "annotation_column": annotation_column,
        "n_train_cells": len(train_indices),
        "source_mtime": source_path.stat().st_mtime if source_path.exists() else None,
        "split_indices_mtime": split_path.stat().st_mtime if split_path.exists() else None,
        "generation_metadata_mtime": generation_path.stat().st_mtime if generation_path.exists() else None,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def build_singler_reference_from_dataset_dir(
    dataset_dir: Path | str,
    annotation_column: str,
    reference_cache_dir: Path | str | None = None,
    force_rebuild: bool = False,
    use_raw: bool = False,
    obs_value_map: dict[str, dict[str, str]] | None = None,
):
    """Reconstruct and cache the train AnnData matching a saved scRNA test dataset."""
    return build_train_reference_from_dataset_dir(
        dataset_dir=dataset_dir,
        annotation_column=annotation_column,
        reference_cache_dir=reference_cache_dir,
        force_rebuild=force_rebuild,
        use_raw=use_raw,
        obs_value_map=obs_value_map,
    )


def _feature_by_cell_matrix(adata) -> Any:
    matrix = adata.X
    if sparse.issparse(matrix):
        return matrix.T.tocsc()
    return np.asarray(matrix, dtype=np.float64).T


def _align_query_and_reference(query, reference) -> tuple[Any, Any, list[str]]:
    query_genes = pd.Index(query.var_names.astype(str))
    reference_genes = pd.Index(reference.var_names.astype(str))
    common_genes = reference_genes[reference_genes.isin(query_genes)].tolist()
    if not common_genes:
        raise ValueError("Query and reference AnnData objects do not share any genes.")

    query_aligned = query[:, common_genes].copy()
    reference_aligned = reference[:, common_genes].copy()
    return query_aligned, reference_aligned, common_genes


def _subsample_reference_by_label(
    reference,
    annotation_column: str,
    max_cells_per_label: int | None,
    random_state: int,
):
    if max_cells_per_label is None:
        return reference
    if max_cells_per_label <= 0:
        raise ValueError("max_reference_cells_per_label must be positive when set.")

    labels = reference.obs[annotation_column].astype(str)
    rng = np.random.default_rng(random_state)
    selected_names: set[str] = set()

    for label in sorted(labels.unique()):
        label_names = labels.index[labels == label].astype(str).to_numpy()
        if len(label_names) > max_cells_per_label:
            label_names = rng.choice(label_names, size=max_cells_per_label, replace=False)
        selected_names.update(map(str, label_names))

    ordered_names = [str(name) for name in reference.obs_names if str(name) in selected_names]
    return reference[ordered_names].copy()


def _extract_column(result: Any, column: str) -> Any:
    if isinstance(result, dict):
        return result[column]
    return result[column]


def _to_list(values: Any) -> list[Any]:
    if hasattr(values, "to_list"):
        return values.to_list()
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def _score_for_predictions(scores: Any, predicted_labels: list[str]) -> list[float] | None:
    if scores is None:
        return None

    if isinstance(scores, pd.DataFrame):
        score_values = []
        for row_index, label in enumerate(predicted_labels):
            if label in scores.columns:
                score_values.append(float(scores.iloc[row_index][label]))
            else:
                score_values.append(float("nan"))
        return score_values

    if hasattr(scores, "column"):
        score_values = []
        for row_index, label in enumerate(predicted_labels):
            try:
                column_values = scores.column(label)
                score_values.append(float(column_values[row_index]))
            except Exception:
                score_values.append(float("nan"))
        return score_values

    try:
        score_frame = pd.DataFrame(scores)
    except Exception:
        return None
    if score_frame.empty:
        return None
    return _score_for_predictions(score_frame, predicted_labels)


def _standardize_singler_results(
    result: Any,
    *,
    query,
    annotation_column: str,
    model_name: str,
) -> pd.DataFrame:
    predicted_labels = [str(label) for label in _to_list(_extract_column(result, "best"))]
    if len(predicted_labels) != query.n_obs:
        raise ValueError(
            f"SingleR returned {len(predicted_labels)} predictions for {query.n_obs} query cells."
        )

    predictions = pd.DataFrame(index=pd.Index(query.obs_names.astype(str), name="cell_id"))
    predictions["true_label"] = query.obs[annotation_column].astype(str).to_numpy()
    predictions["predicted_label"] = predicted_labels

    try:
        scores = _extract_column(result, "scores")
    except Exception:
        scores = None
    score_values = _score_for_predictions(scores, predicted_labels)
    if score_values is not None:
        predictions["score"] = score_values

    predictions["model_name"] = model_name
    return predictions


def run_singler_annotation(
    config: SingleRAnnotationConfig,
    singler_module: Any | None = None,
) -> pd.DataFrame:
    """Run Python SingleR on a test AnnData object using a matching train reference."""
    sc = _import_scanpy_dependency()
    singler = singler_module or _import_singler_dependency()

    adata_path = Path(config.adata_path)
    if not adata_path.exists():
        raise FileNotFoundError(f"AnnData test file not found: {adata_path}")

    query = sc.read_h5ad(adata_path)
    if config.annotation_column not in query.obs:
        raise ValueError(
            f"Annotation column {config.annotation_column!r} not found in query adata.obs."
        )

    if config.reference_adata_path is not None:
        reference_path = Path(config.reference_adata_path)
        if not reference_path.exists():
            raise FileNotFoundError(f"SingleR reference AnnData file not found: {reference_path}")
        reference = sc.read_h5ad(reference_path)
    elif config.dataset_dir is not None:
        reference = build_singler_reference_from_dataset_dir(
            dataset_dir=config.dataset_dir,
            annotation_column=config.annotation_column,
            reference_cache_dir=config.reference_cache_dir,
            force_rebuild=config.force_rebuild_reference,
            use_raw=config.reference_use_raw,
            obs_value_map=config.reference_obs_value_map,
        )
    else:
        raise ValueError("Either reference_adata_path or dataset_dir must be provided.")

    if config.annotation_column not in reference.obs:
        raise ValueError(
            f"Annotation column {config.annotation_column!r} not found in reference adata.obs."
        )
    reference = _subsample_reference_by_label(
        reference=reference,
        annotation_column=config.annotation_column,
        max_cells_per_label=config.max_reference_cells_per_label,
        random_state=config.reference_sample_seed,
    )

    query_aligned, reference_aligned, genes = _align_query_and_reference(query, reference)
    result = singler.annotate_single(
        test_data=_feature_by_cell_matrix(query_aligned),
        ref_data=_feature_by_cell_matrix(reference_aligned),
        ref_labels=reference_aligned.obs[config.annotation_column].astype(str).to_numpy(),
        test_features=genes,
        ref_features=genes,
        train_args=config.train_args or {},
        classify_args=config.classify_args or {},
        num_threads=int(config.num_threads),
    )
    return _standardize_singler_results(
        result,
        query=query_aligned,
        annotation_column=config.annotation_column,
        model_name=config.model_name,
    )
