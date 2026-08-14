from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap

from alias.evaluation.embedding import load_dataset_embedding_artifacts
from alias.util.artifacts import create_evaluation_run_directory, write_metadata
from alias.util.plots.umap_plots import UMAPCellPlotter


@dataclass
class EmbeddingSubsetUMAPConfig:
    output_dir: Path | str
    annotation_column: str
    subset_column: str | None = None
    subsets: Mapping[str, list[str] | set[str] | tuple[str, ...]] | None = None
    color_columns: list[str] = field(default_factory=list)
    metadata_columns: list[str] = field(default_factory=list)
    include_global: bool = True
    include_centroids: bool = True
    annotate_centroids: bool = True
    n_neighbors: int = 15
    min_dist: float = 0.5
    n_components: int = 50
    random_state: int = 73
    run_timestamp: str | None = None


def _metadata_frame(metadata: Any | None) -> pd.DataFrame | None:
    if metadata is None:
        return None
    if isinstance(metadata, pd.DataFrame):
        df = metadata.copy()
    elif hasattr(metadata, "obs"):
        df = metadata.obs.copy()
    else:
        raise TypeError("metadata must be a pandas DataFrame, AnnData-like object, or None.")
    df.index = df.index.astype(str)
    return df


def _merge_cell_metadata(
    df_cells: pd.DataFrame,
    metadata: pd.DataFrame | None,
    metadata_columns: list[str],
) -> pd.DataFrame:
    df = df_cells.copy()
    df.index = df.index.astype(str)
    if metadata is None or not metadata_columns:
        return df

    missing_columns = [column for column in metadata_columns if column not in metadata.columns]
    if missing_columns:
        raise ValueError(f"Metadata is missing requested columns: {missing_columns}")

    metadata_subset = metadata.loc[:, metadata_columns].copy()
    metadata_subset.index = metadata_subset.index.astype(str)
    for column in metadata_columns:
        values = metadata_subset[column].reindex(df.index)
        missing_fraction = float(values.isna().mean()) if len(values) else 0.0
        if missing_fraction > 0.05:
            raise ValueError(
                f"Metadata column `{column}` could not be aligned for "
                f"{missing_fraction:.1%} of cells. Check embedding cell IDs against metadata index."
            )
        df[column] = values
    return df


def _fill_time_from_label(df_cells: pd.DataFrame) -> pd.DataFrame:
    df = df_cells.copy()
    if "label" not in df.columns:
        return df

    extracted = df["label"].astype(str).str.extract(r"_([0-9]+(?:\.[0-9]+)?)$")[0]
    extracted = pd.to_numeric(extracted, errors="coerce")
    if "time" not in df.columns:
        df["time"] = extracted
    else:
        df["time"] = pd.to_numeric(df["time"], errors="coerce").fillna(extracted)
    return df


def _add_subset_column(
    df_cells: pd.DataFrame,
    annotation_column: str,
    subset_column: str | None,
    subsets: Mapping[str, list[str] | set[str] | tuple[str, ...]] | None,
) -> pd.DataFrame:
    df = df_cells.copy()
    if subset_column is None or subsets is None:
        return df

    label_to_subset: dict[str, str] = {}
    for subset_name, labels in subsets.items():
        for label in labels:
            label_to_subset.setdefault(str(label), subset_name)

    df[subset_column] = df[annotation_column].astype(str).map(label_to_subset)
    df[subset_column] = df[subset_column].where(df[subset_column].notna(), pd.NA)
    return df


def _is_embedding_column(column: Any) -> bool:
    if isinstance(column, (int, np.integer)):
        return True
    if isinstance(column, str) and column.isdigit():
        return True
    return False


def _embedding_columns(df: pd.DataFrame, excluded_columns: set[str]) -> list[Any]:
    return [
        column
        for column in df.columns
        if column not in excluded_columns
        and _is_embedding_column(column)
        and np.issubdtype(df[column].dtype, np.number)
    ]


def _ensure_centroid_annotation_column(df_centroids: pd.DataFrame, annotation_column: str) -> pd.DataFrame:
    df = df_centroids.copy()
    if annotation_column not in df.columns:
        if "cell_type" in df.columns:
            df[annotation_column] = df["cell_type"].astype(str)
        else:
            df[annotation_column] = df.index.astype(str)
    return df


def _finalize_centroids_for_plotting(
    df_centroids_umap: pd.DataFrame,
    annotation_column: str,
) -> pd.DataFrame:
    df = _ensure_centroid_annotation_column(df_centroids_umap, annotation_column)
    return (
        df.loc[:, [annotation_column, "UMAP1", "UMAP2"]]
        .rename(columns={annotation_column: "cell_type"})
        .reset_index(drop=True)
    )


def _compute_embedding_umap(df: pd.DataFrame, config: EmbeddingSubsetUMAPConfig) -> pd.DataFrame:
    embedding_array = np.vstack(df["embedding"].values).astype(np.float32)
    if embedding_array.shape[0] < 3:
        raise ValueError("At least three rows are required to compute a subset UMAP.")

    n_pca = min(config.n_components, embedding_array.shape[0] - 1, embedding_array.shape[1])
    if n_pca >= 2:
        embedding_array = PCA(
            n_components=n_pca,
            random_state=config.random_state,
            svd_solver="randomized",
        ).fit_transform(embedding_array)

    n_neighbors = min(config.n_neighbors, embedding_array.shape[0] - 1)
    n_neighbors = max(2, n_neighbors)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=config.min_dist,
        random_state=config.random_state,
    )
    coords = reducer.fit_transform(embedding_array)

    result = df.copy()
    result["UMAP1"] = coords[:, 0]
    result["UMAP2"] = coords[:, 1]
    return result


def _plot_color_column(
    df_cells_umap: pd.DataFrame,
    df_centroids_umap: pd.DataFrame | None,
    color_column: str,
    output_dir: Path,
    subset_name: str,
    annotation_column: str,
    annotate_centroids: bool,
) -> None:
    if color_column not in df_cells_umap.columns:
        return

    plotter = UMAPCellPlotter()
    plotter.annotate_centroids = False

    if pd.api.types.is_numeric_dtype(df_cells_umap[color_column]):
        color_values = pd.to_numeric(df_cells_umap[color_column], errors="coerce")
        color_kwargs = {
            "time_color_column": color_column,
            "vmin": float(color_values.min()),
            "vmax": float(color_values.max()),
        }
    else:
        color_kwargs = {"annotation_column": color_column}

    plotter.plot_cells(
        df_cells_umap,
        output_path=output_dir / f"cells_colored_by_{color_column}_{subset_name}.pdf",
        title=f"Cells Colored by {color_column} - {subset_name}",
        **color_kwargs,
    )

    if annotate_centroids and df_centroids_umap is not None:
        plotter.annotate_centroids = True
        plotter.plot_cells(
            df_cells_umap,
            output_path=output_dir / f"cells_colored_by_{color_column}_with_labels_{subset_name}.pdf",
            annotate_centroids_df=df_centroids_umap,
            title=f"Cells Colored by {color_column} with Labels - {subset_name}",
            **color_kwargs,
        )


def _prepare_combined_umap_input(
    df_cells: pd.DataFrame,
    df_centroids: pd.DataFrame | None,
    annotation_column: str,
    include_centroids: bool,
) -> tuple[pd.DataFrame, int]:
    cell_embedding_columns = _embedding_columns(df_cells, set())
    df_cells_combined = df_cells.copy()
    df_cells_combined["embedding"] = df_cells[cell_embedding_columns].to_numpy(dtype=np.float32).tolist()

    if not include_centroids or df_centroids is None or df_centroids.empty:
        return df_cells_combined, len(df_cells_combined)

    excluded_centroid_columns = {annotation_column, "cell_type"}
    centroid_embedding_columns = _embedding_columns(df_centroids, excluded_centroid_columns)
    df_centroids_combined = df_centroids.copy()
    df_centroids_combined["embedding"] = df_centroids[centroid_embedding_columns].to_numpy(dtype=np.float32).tolist()

    return pd.concat([df_cells_combined, df_centroids_combined], axis=0), len(df_cells_combined)


def _default_color_columns(config: EmbeddingSubsetUMAPConfig) -> list[str]:
    columns = list(config.color_columns)
    if not columns:
        columns = [config.annotation_column]
        columns.extend(config.metadata_columns)
        if config.subset_column is not None:
            columns.append(config.subset_column)
    return list(dict.fromkeys(columns))


def embedding_subset_umap_plots(
    embeddings_dict: dict,
    config: EmbeddingSubsetUMAPConfig,
    metadata: Any | None = None,
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    metadata_df = _metadata_frame(metadata)
    output_root = Path(config.output_dir)
    subset_umaps: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}

    for model_name, model_data in embeddings_dict.items():
        subset_umaps[model_name] = {}
        for dataset_name, dataset_meta in model_data.items():
            loaded_dataset = load_dataset_embedding_artifacts(
                dataset_meta,
                annotation_column=config.annotation_column,
            )
            loaded_artifacts = loaded_dataset["artifacts"]
            if "df_cells" not in loaded_artifacts:
                continue

            df_cells = loaded_artifacts["df_cells"]["dataframe"].copy()
            df_cells = _merge_cell_metadata(df_cells, metadata_df, config.metadata_columns)
            if "time" in config.metadata_columns or "time" in config.color_columns:
                df_cells = _fill_time_from_label(df_cells)
            df_cells = _add_subset_column(
                df_cells,
                annotation_column=config.annotation_column,
                subset_column=config.subset_column,
                subsets=config.subsets,
            )

            df_centroids = None
            if "df_celltypes" in loaded_artifacts:
                df_centroids = loaded_artifacts["df_celltypes"]["dataframe"].copy()
                df_centroids = _ensure_centroid_annotation_column(df_centroids, config.annotation_column)

            run_dir = create_evaluation_run_directory(
                output_dir=output_root,
                model_name=model_name,
                dataset_name=dataset_name,
                evaluation_name="embedding_subset_umaps",
                timestamp=config.run_timestamp,
            )

            subset_specs: list[tuple[str, pd.DataFrame]] = []
            if config.include_global:
                subset_specs.append(("all_cells", df_cells))
            if config.subset_column is not None:
                for subset_name in sorted(df_cells[config.subset_column].dropna().unique()):
                    subset_df = df_cells[df_cells[config.subset_column] == subset_name].copy()
                    if not subset_df.empty:
                        subset_specs.append((str(subset_name), subset_df))

            subset_umaps[model_name][dataset_name] = {}
            for subset_name, subset_cells in subset_specs:
                subset_dir = run_dir / subset_name
                subset_dir.mkdir(parents=True, exist_ok=True)

                subset_centroids = None
                if df_centroids is not None:
                    labels = subset_cells[config.annotation_column].astype(str).unique().tolist()
                    subset_centroids = df_centroids[df_centroids[config.annotation_column].isin(labels)].copy()

                combined, n_cells = _prepare_combined_umap_input(
                    subset_cells,
                    subset_centroids,
                    annotation_column=config.annotation_column,
                    include_centroids=config.include_centroids,
                )
                df_umap = _compute_embedding_umap(combined, config)
                df_cells_umap = df_umap.iloc[:n_cells].copy()
                df_centroids_umap = df_umap.iloc[n_cells:].copy() if len(df_umap) > n_cells else None
                if df_centroids_umap is not None:
                    df_centroids_umap = _finalize_centroids_for_plotting(
                        df_centroids_umap,
                        config.annotation_column,
                    )

                cells_path = subset_dir / "cell_umap_coordinates.parquet"
                df_cells_umap.to_parquet(cells_path)
                centroids_path = None
                if df_centroids_umap is not None:
                    centroids_path = subset_dir / "centroid_umap_coordinates.parquet"
                    df_centroids_umap.to_parquet(centroids_path)

                for color_column in _default_color_columns(config):
                    _plot_color_column(
                        df_cells_umap=df_cells_umap,
                        df_centroids_umap=df_centroids_umap,
                        color_column=color_column,
                        output_dir=subset_dir,
                        subset_name=subset_name,
                        annotation_column=config.annotation_column,
                        annotate_centroids=config.annotate_centroids,
                    )

                subset_umaps[model_name][dataset_name][subset_name] = {
                    "cells": df_cells_umap,
                    "centroids": df_centroids_umap,
                    "cell_umap_path": str(cells_path),
                    "centroid_umap_path": str(centroids_path) if centroids_path is not None else None,
                    "subset_dir": str(subset_dir),
                }

            write_metadata(
                run_dir,
                {
                    "evaluation_name": "embedding_subset_umaps",
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "run_timestamp": run_dir.name,
                    "embedding_run_timestamp": loaded_dataset["run_timestamp"],
                    "annotation_column": config.annotation_column,
                    "subset_column": config.subset_column,
                    "subsets": {
                        key: sorted(map(str, values))
                        for key, values in (config.subsets or {}).items()
                    },
                    "color_columns": _default_color_columns(config),
                    "umap_settings": {
                        "n_neighbors": config.n_neighbors,
                        "min_dist": config.min_dist,
                        "n_components": config.n_components,
                        "random_state": config.random_state,
                    },
                },
            )

    return subset_umaps
