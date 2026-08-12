from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from alias.evaluation.embedding import clean_model_name, load_dataset_embedding_artifacts


def _resolve_model_key(embeddings_dict: dict[str, Any], model_key: str | None) -> str:
    if model_key is None:
        if len(embeddings_dict) != 1:
            raise ValueError(
                "model_key must be provided when embeddings_dict contains multiple models."
            )
        return next(iter(embeddings_dict))

    if model_key in embeddings_dict:
        return model_key

    cleaned_key = clean_model_name(str(model_key))
    if cleaned_key in embeddings_dict:
        return cleaned_key

    raise KeyError(f"Model {model_key!r} not found in embeddings_dict.")


def _resolve_dataset_meta(
    embeddings_dict: dict[str, Any] | None,
    dataset_meta: dict[str, Any] | None,
    model_key: str | None,
    dataset_key: str,
) -> tuple[str, dict[str, Any]]:
    if dataset_meta is not None:
        return clean_model_name(str(model_key or "embedding_model")), dataset_meta

    if embeddings_dict is None:
        raise ValueError("Provide either embeddings_dict or dataset_meta.")

    resolved_model_key = _resolve_model_key(embeddings_dict, model_key)
    model_meta = embeddings_dict[resolved_model_key]
    if dataset_key not in model_meta:
        raise KeyError(
            f"Dataset {dataset_key!r} not found for model {resolved_model_key!r}."
        )

    return resolved_model_key, model_meta[dataset_key]


def _load_cell_artifacts(
    dataset_meta: dict[str, Any],
    annotation_column: str | None,
) -> dict[str, Any]:
    if "artifacts" in dataset_meta:
        artifacts = dataset_meta["artifacts"]
    else:
        artifacts = load_dataset_embedding_artifacts(
            dataset_meta,
            annotation_column=annotation_column,
        )["artifacts"]

    if "df_cells" not in artifacts:
        raise KeyError("Embedding metadata does not contain a df_cells artifact.")

    return artifacts["df_cells"]


def _embedding_columns(df: pd.DataFrame) -> list[Any]:
    excluded = {"UMAP1", "UMAP2"}
    columns = [
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not columns:
        raise ValueError("No numeric embedding columns found in df_cells artifact.")
    return columns


def _copy_or_subset_adata(
    adata: Any,
    frame: pd.DataFrame,
    require_all_cells: bool,
    copy: bool,
) -> tuple[Any, pd.DataFrame]:
    aligned_frame = frame.copy()
    aligned_frame.index = aligned_frame.index.astype(str)
    if aligned_frame.index.has_duplicates:
        duplicates = aligned_frame.index[aligned_frame.index.duplicated()].unique()
        raise ValueError(
            "Embedding artifact contains duplicate cell ids after string conversion: "
            f"{duplicates[:5].tolist()}"
        )

    obs_names = pd.Index(adata.obs_names.astype(str))
    shared_mask = obs_names.isin(aligned_frame.index)

    if not shared_mask.any():
        raise ValueError("No shared cell ids between AnnData observations and embeddings.")

    missing_count = int((~shared_mask).sum())
    if missing_count and require_all_cells:
        raise ValueError(
            f"{missing_count} AnnData observations do not have matching embeddings."
        )
    if missing_count and not copy:
        raise ValueError(
            "copy=False requires embeddings for every AnnData observation. "
            f"{missing_count} observations are missing."
        )

    if missing_count:
        positions = np.flatnonzero(shared_mask)
        adata_out = adata[positions].copy()
        obs_names = obs_names[shared_mask]
    else:
        adata_out = adata.copy() if copy else adata

    frame_out = aligned_frame.loc[obs_names]
    return adata_out, frame_out


def add_embeddings_to_adata(
    adata: Any,
    embeddings_dict: dict[str, Any] | None = None,
    *,
    dataset_meta: dict[str, Any] | None = None,
    model_key: str | None = None,
    dataset_key: str = "scrna",
    annotation_column: str | None = None,
    obsm_key: str | None = None,
    copy: bool = True,
    require_all_cells: bool = False,
) -> Any:
    """Add saved cell embeddings to ``adata.obsm`` aligned by observation ids.

    Parameters
    ----------
    adata
        AnnData-like object with ``obs_names`` and ``obsm``.
    embeddings_dict
        Nested output from ``generate_embeddings`` or ``umap_plots``.
    dataset_meta
        Optional metadata for one dataset, for example ``embeddings_dict[model]["scrna"]``.
    model_key
        Model key in ``embeddings_dict``. If omitted, ``embeddings_dict`` must contain
        exactly one model.
    dataset_key
        Dataset key inside the model metadata, usually ``"scrna"``.
    annotation_column
        Optional annotation column to restore while loading artifacts.
    obsm_key
        Destination key. Defaults to ``X_<model_key>``.
    copy
        Return a copied AnnData object. If ``False``, every observation must have an
        embedding because AnnData ``obsm`` arrays must match ``n_obs``.
    require_all_cells
        Raise if any AnnData observation is missing from the embedding artifact.
    """
    resolved_model_key, resolved_dataset_meta = _resolve_dataset_meta(
        embeddings_dict,
        dataset_meta,
        model_key,
        dataset_key,
    )
    cell_artifact = _load_cell_artifacts(resolved_dataset_meta, annotation_column)
    df_cells = cell_artifact["dataframe"]
    embedding_columns = _embedding_columns(df_cells)

    adata_out, aligned = _copy_or_subset_adata(
        adata,
        df_cells,
        require_all_cells=require_all_cells,
        copy=copy,
    )
    destination = obsm_key or f"X_{resolved_model_key}"
    adata_out.obsm[destination] = aligned[embedding_columns].to_numpy(dtype=np.float32)
    return adata_out


def add_umap_to_adata(
    adata: Any,
    embeddings_dict: dict[str, Any] | None = None,
    *,
    dataset_meta: dict[str, Any] | None = None,
    model_key: str | None = None,
    dataset_key: str = "scrna",
    annotation_column: str | None = None,
    obsm_key: str | None = None,
    copy: bool = True,
    require_all_cells: bool = False,
) -> Any:
    """Add saved UMAP coordinates from a cell embedding artifact to ``adata.obsm``."""
    resolved_model_key, resolved_dataset_meta = _resolve_dataset_meta(
        embeddings_dict,
        dataset_meta,
        model_key,
        dataset_key,
    )
    cell_artifact = _load_cell_artifacts(resolved_dataset_meta, annotation_column)
    umap_df = cell_artifact.get("umap")
    if umap_df is None:
        raise ValueError("df_cells artifact does not include saved UMAP coordinates.")
    if not {"UMAP1", "UMAP2"}.issubset(umap_df.columns):
        raise ValueError("Saved UMAP artifact must contain UMAP1 and UMAP2 columns.")

    adata_out, aligned = _copy_or_subset_adata(
        adata,
        umap_df,
        require_all_cells=require_all_cells,
        copy=copy,
    )
    destination = obsm_key or f"X_umap_{resolved_model_key}"
    adata_out.obsm[destination] = aligned[["UMAP1", "UMAP2"]].to_numpy(dtype=np.float32)
    return adata_out


def add_embedding_artifacts_to_adata(
    adata: Any,
    embeddings_dict: dict[str, Any] | None = None,
    *,
    dataset_meta: dict[str, Any] | None = None,
    model_key: str | None = None,
    dataset_key: str = "scrna",
    annotation_column: str | None = None,
    embedding_obsm_key: str | None = None,
    umap_obsm_key: str | None = None,
    include_umap: bool = True,
    copy: bool = True,
    require_all_cells: bool = False,
) -> Any:
    """Add both full embeddings and saved UMAP coordinates to ``adata.obsm``."""
    adata_out = add_embeddings_to_adata(
        adata,
        embeddings_dict,
        dataset_meta=dataset_meta,
        model_key=model_key,
        dataset_key=dataset_key,
        annotation_column=annotation_column,
        obsm_key=embedding_obsm_key,
        copy=copy,
        require_all_cells=require_all_cells,
    )

    if include_umap:
        adata_out = add_umap_to_adata(
            adata_out,
            embeddings_dict,
            dataset_meta=dataset_meta,
            model_key=model_key,
            dataset_key=dataset_key,
            annotation_column=annotation_column,
            obsm_key=umap_obsm_key,
            copy=False,
            require_all_cells=True,
        )

    return adata_out


def add_umaps_to_adata(
    adata: Any,
    embeddings_dict: dict[str, Any],
    *,
    dataset_key: str = "scrna",
    annotation_column: str | None = None,
    model_keys: list[str] | None = None,
    copy: bool = True,
) -> Any:
    """Legacy-compatible helper that adds saved UMAPs for one or more models.

    Older manuscript scripts used ``add_umaps_to_adata(adata, embeddings_dict)``.
    This wrapper keeps that workflow while the more explicit ``add_umap_to_adata``
    and ``add_embedding_artifacts_to_adata`` helpers are available for new scripts.
    """
    selected_model_keys = model_keys or list(embeddings_dict.keys())
    adata_out = adata.copy() if copy else adata

    for index, model_key in enumerate(selected_model_keys):
        adata_out = add_umap_to_adata(
            adata_out,
            embeddings_dict,
            model_key=model_key,
            dataset_key=dataset_key,
            annotation_column=annotation_column,
            copy=index == 0 and copy,
            require_all_cells=index > 0,
        )

    return adata_out
