from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INDEX_COLUMNS = ("index", "cell_id", "obs_name", "barcode")
PYARROW_INSTALL_COMMAND = (
    'uv pip install --python .venv_annotation/bin/python "pyarrow>=15"'
)


def _read_embedding_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except ImportError as error:
            raise ImportError(
                "pyarrow is required to read parquet embedding files. Install it with:\n"
                f"{PYARROW_INSTALL_COMMAND}"
            ) from error
    raise ValueError(
        f"Unsupported external embedding file format: {path}. "
        "Expected .csv or .parquet."
    )


def load_external_embedding_frame(
    path: str | Path,
    *,
    index_column: str | None = None,
) -> pd.DataFrame:
    """Load precomputed external embeddings with cell ids as the index."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"External embedding file not found: {path}")

    df = _read_embedding_table(path)
    resolved_index_column = index_column
    if resolved_index_column is None:
        resolved_index_column = next(
            (column for column in DEFAULT_INDEX_COLUMNS if column in df.columns),
            None,
        )

    if resolved_index_column is not None and resolved_index_column in df.columns:
        df = df.set_index(resolved_index_column)

    df.index = df.index.astype(str)
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        raise ValueError(f"No numeric embedding columns found in {path}")
    return numeric_df.astype(np.float32)


def align_embedding_frame_to_adata(
    adata: Any,
    embeddings: pd.DataFrame,
    *,
    obsm_key: str,
    copy: bool = True,
    require_all_cells: bool = True,
) -> Any:
    """Align a cell-indexed embedding frame to AnnData observations and store it in obsm."""
    adata_out = adata.copy() if copy else adata
    embeddings = embeddings.copy()
    embeddings.index = embeddings.index.astype(str)
    obs_names = adata_out.obs_names.astype(str)

    missing = obs_names.difference(embeddings.index)
    if require_all_cells and len(missing):
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"{obsm_key} is missing embeddings for {len(missing)} AnnData cells. "
            f"First missing ids: {preview}"
        )

    if len(missing):
        keep_mask = obs_names.isin(embeddings.index)
        adata_out = adata_out[keep_mask].copy()
        obs_names = adata_out.obs_names.astype(str)

    aligned = embeddings.loc[obs_names]
    adata_out.obsm[obsm_key] = aligned.to_numpy(dtype=np.float32)
    return adata_out


def require_optional_dependency(
    module_name: str,
    *,
    install_command: str,
    package_label: str | None = None,
) -> Any:
    """Import an optional package with a project-specific install hint."""
    try:
        return importlib.import_module(module_name)
    except (ImportError, OSError) as error:
        label = package_label or module_name
        raise ImportError(
            f"{label} is required for this embedding generator. Install it with:\n"
            f"{install_command}\n"
            f"Underlying import error: {type(error).__name__}: {error}"
        ) from error


def embedding_frame_from_matrix(
    cell_ids: list[str] | pd.Index,
    embeddings: np.ndarray,
    *,
    index_column: str = "index",
) -> pd.DataFrame:
    """Create the external embedding table format used by the scIB loaders."""
    embedding_array = np.asarray(embeddings, dtype=np.float32)
    if embedding_array.ndim != 2:
        raise ValueError(
            "External embeddings must be a 2-dimensional matrix, got "
            f"shape {embedding_array.shape}."
        )

    cell_id_list = [str(cell_id) for cell_id in cell_ids]
    if len(cell_id_list) != embedding_array.shape[0]:
        raise ValueError(
            "Number of cell ids does not match embedding rows: "
            f"{len(cell_id_list)} ids for {embedding_array.shape[0]} rows."
        )

    embedding_columns = [str(index) for index in range(embedding_array.shape[1])]
    frame = pd.DataFrame(embedding_array, columns=embedding_columns)
    frame.insert(0, index_column, cell_id_list)
    return frame


def _write_embedding_frame(frame: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
        return
    if path.suffix in {".parquet", ".pq"}:
        try:
            frame.to_parquet(path, index=False)
        except ImportError as error:
            raise ImportError(
                "pyarrow is required to write parquet embedding files. Install it with:\n"
                f"{PYARROW_INSTALL_COMMAND}"
            ) from error
        return
    raise ValueError(
        f"Unsupported external embedding file format: {path}. "
        "Expected .csv or .parquet."
    )


def save_external_embedding_artifacts(
    *,
    output_path: str | Path,
    metadata_path: str | Path,
    cell_ids: list[str] | pd.Index,
    embeddings: np.ndarray,
    metadata: dict[str, Any],
    force: bool = False,
    index_column: str = "index",
) -> tuple[Path, Path]:
    """Save external embeddings and metadata in the shared benchmark format."""
    output_path = Path(output_path)
    metadata_path = Path(metadata_path)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Embedding output already exists: {output_path}. Pass force=True to overwrite."
        )

    frame = embedding_frame_from_matrix(
        cell_ids,
        embeddings,
        index_column=index_column,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    _write_embedding_frame(frame, output_path)

    enriched_metadata = {
        **metadata,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
        "index_column": index_column,
        "n_obs": int(frame.shape[0]),
        "embedding_dim": int(frame.shape[1] - 1),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(enriched_metadata, handle, indent=2, sort_keys=True)
    return output_path, metadata_path
