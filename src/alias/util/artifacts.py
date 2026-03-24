from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _sanitize_component(value: str) -> str:
    """Return a filesystem-safe path component."""
    cleaned = re.sub(r"\s+", "_", str(value).strip())
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "", cleaned)
    return cleaned or "unknown"


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")


def _unique_run_directory(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir

    suffix = 1
    while True:
        candidate = base_dir.with_name(f"{base_dir.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate
        suffix += 1


def create_run_directory(
    root_dir: Path | str,
    category: str,
    dataset_name: str,
    model_name: str | None = None,
    evaluation_name: str | None = None,
    timestamp: str | None = None,
) -> Path:
    """Create and return a timestamped run directory."""
    root_path = Path(root_dir)
    parts = [_sanitize_component(dataset_name)]

    if category:
        parts.insert(0, _sanitize_component(category))

    if model_name:
        parts.append(_sanitize_component(model_name))
    if evaluation_name:
        parts.append(_sanitize_component(evaluation_name))

    parts.append(_sanitize_component(timestamp or _timestamp_now()))

    run_dir = _unique_run_directory(root_path.joinpath(*parts))
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def create_evaluation_run_directory(
    output_dir: Path | str,
    model_name: str,
    dataset_name: str,
    evaluation_name: str,
    timestamp: str | None = None,
) -> Path:
    """Create a timestamped evaluation run using the canonical evaluation layout."""
    return create_run_directory(
        root_dir=Path(output_dir) / _sanitize_component(model_name),
        category=dataset_name,
        dataset_name=evaluation_name,
        timestamp=timestamp,
    )


def write_metadata(run_dir: Path | str, metadata: dict[str, Any]) -> Path:
    """Write metadata.json in the provided run directory."""
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    metadata_path = run_path / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    return metadata_path


def save_embedding_frame(
    output_dir: Path | str,
    artifact_name: str,
    embedding_df: pd.DataFrame,
    annotation_map: dict[str, Any] | None = None,
    annotation_file_name: str | None = None,
) -> dict[str, Any]:
    """Save one embedding dataframe plus optional annotation metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    parquet_path = output_path / f"{artifact_name}.parquet"
    embedding_df.to_parquet(parquet_path)

    annotation_path = None
    if annotation_map is not None:
        annotation_path = output_path / (
            annotation_file_name or f"{artifact_name}_annotations.json"
        )
        with annotation_path.open("w", encoding="utf-8") as handle:
            json.dump(annotation_map, handle, indent=2, sort_keys=True)

    return {
        "path": str(parquet_path),
        "annotation_map": str(annotation_path) if annotation_path is not None else None,
        "n_samples": len(embedding_df),
        "embedding_dim": int(embedding_df.shape[1]) if not embedding_df.empty else 0,
    }


def load_embedding_frame(path: Path | str) -> pd.DataFrame:
    """Load a saved embedding parquet file."""
    return pd.read_parquet(path)


def load_annotation_map(path: Path | str | None) -> dict[str, Any] | None:
    """Load optional JSON annotation metadata."""
    if path is None:
        return None

    annotation_path = Path(path)
    if not annotation_path.exists():
        return None

    with annotation_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
