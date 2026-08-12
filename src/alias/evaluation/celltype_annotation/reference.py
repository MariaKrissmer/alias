from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any


@dataclass
class TrainReferenceMetadata:
    dataset_dir: Path
    source_path: Path
    split_indices_path: Path
    generation_metadata_path: Path
    annotation_column: str
    train_indices: list[str]
    scrna_config: Any
    use_raw: bool = False
    obs_value_map: dict[str, dict[str, str]] | None = None

    def cache_fingerprint(self) -> dict[str, Any]:
        return {
            "dataset_dir": str(self.dataset_dir),
            "source_path": str(self.source_path),
            "split_indices_path": str(self.split_indices_path),
            "generation_metadata_path": str(self.generation_metadata_path),
            "annotation_column": self.annotation_column,
            "n_train_cells": len(self.train_indices),
            "use_raw": self.use_raw,
            "obs_value_map": self.obs_value_map or {},
            "source_mtime": (
                self.source_path.stat().st_mtime if self.source_path.exists() else None
            ),
            "split_indices_mtime": (
                self.split_indices_path.stat().st_mtime
                if self.split_indices_path.exists()
                else None
            ),
            "generation_metadata_mtime": (
                self.generation_metadata_path.stat().st_mtime
                if self.generation_metadata_path.exists()
                else None
            ),
        }


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def scrna_config_from_metadata(config_dict: dict[str, Any]):
    from alias.data.scrna import DatascRNAConfig

    allowed = {field.name for field in fields(DatascRNAConfig)}
    clean = {key: value for key, value in config_dict.items() if key in allowed}
    return DatascRNAConfig(**clean)


def load_train_reference_metadata(
    dataset_dir: Path | str,
    annotation_column: str,
    use_raw: bool = False,
    obs_value_map: dict[str, dict[str, str]] | None = None,
) -> TrainReferenceMetadata:
    dataset_path = Path(dataset_dir)
    generation_path = dataset_path / "metadata" / "generation_metadata.json"
    split_path = dataset_path / "metadata" / "split_indices.json"
    if not generation_path.exists():
        raise FileNotFoundError(f"Missing generation metadata: {generation_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split indices: {split_path}")

    generation_metadata = read_json(generation_path)
    split_metadata = read_json(split_path)
    train_indices = [str(index) for index in split_metadata.get("train_indices", [])]
    if not train_indices:
        raise ValueError(f"No train_indices found in {split_path}")

    scrna_config_dict = generation_metadata.get("scrna_config", {})
    scrna_config = scrna_config_from_metadata(scrna_config_dict)
    scrna_config.annotation_column = annotation_column
    source = generation_metadata.get("source") or scrna_config_dict.get("source")
    if not source:
        raise ValueError(f"No source AnnData path recorded in {generation_path}")
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source AnnData file not found: {source_path}")

    return TrainReferenceMetadata(
        dataset_dir=dataset_path,
        source_path=source_path,
        split_indices_path=split_path,
        generation_metadata_path=generation_path,
        annotation_column=annotation_column,
        train_indices=train_indices,
        scrna_config=scrna_config,
        use_raw=use_raw,
        obs_value_map=obs_value_map,
    )


def cache_metadata_matches(metadata_path: Path, fingerprint: dict[str, Any]) -> bool:
    if not metadata_path.exists():
        return False

    try:
        metadata = read_json(metadata_path)
    except (OSError, json.JSONDecodeError):
        return False

    return all(metadata.get(key) == value for key, value in fingerprint.items())


def build_train_reference_from_dataset_dir(
    dataset_dir: Path | str,
    annotation_column: str,
    reference_cache_dir: Path | str | None = None,
    force_rebuild: bool = False,
    use_raw: bool = False,
    obs_value_map: dict[str, dict[str, str]] | None = None,
):
    """Reconstruct and cache train AnnData matching a saved scRNA test dataset."""
    import scanpy as sc
    from alias.data.scrna import _prepare_adata_for_scrna

    metadata = load_train_reference_metadata(
        dataset_dir=dataset_dir,
        annotation_column=annotation_column,
        use_raw=use_raw,
        obs_value_map=obs_value_map,
    )
    cache_dir = (
        Path(reference_cache_dir)
        if reference_cache_dir is not None
        else metadata.dataset_dir / "celltype_annotation" / "references"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "adata_train_reference.h5ad"
    cache_metadata_path = cache_dir / "adata_train_reference_metadata.json"
    fingerprint = metadata.cache_fingerprint()

    if (
        not force_rebuild
        and cache_path.exists()
        and cache_metadata_matches(cache_metadata_path, fingerprint)
    ):
        return sc.read_h5ad(cache_path)

    adata = sc.read_h5ad(metadata.source_path)
    for obs_column, replacements in (obs_value_map or {}).items():
        if obs_column not in adata.obs:
            raise ValueError(f"Cannot map missing obs column {obs_column!r}.")
        adata.obs[obs_column] = adata.obs[obs_column].replace(replacements)
    if use_raw:
        if adata.raw is None:
            raise ValueError(f"Requested raw reference from {metadata.source_path}, but adata.raw is missing.")
        adata = adata.raw.to_adata()
    adata.obs_names_make_unique()
    prepared = _prepare_adata_for_scrna(adata, metadata.scrna_config)
    missing_indices = sorted(set(metadata.train_indices) - set(prepared.obs_names.astype(str)))
    if missing_indices:
        preview = ", ".join(missing_indices[:5])
        raise ValueError(
            f"{len(missing_indices)} train indices are missing after preprocessing; "
            f"first missing indices: {preview}"
        )
    if annotation_column not in prepared.obs:
        raise ValueError(f"Annotation column {annotation_column!r} not found in reference obs.")

    reference = prepared[metadata.train_indices].copy()
    reference.write_h5ad(cache_path)
    write_json(cache_metadata_path, fingerprint)
    return reference
