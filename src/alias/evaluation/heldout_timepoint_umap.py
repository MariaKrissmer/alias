from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json
import re

import numpy as np
import pandas as pd

from alias.evaluation.celltype_label_plots import EvaluationConfig, compute_umap, get_umap_plotter_class
from alias.util.artifacts import create_evaluation_run_directory, write_metadata


def _safe_component(value: Any) -> str:
    cleaned = re.sub(r"\s+", "_", str(value).strip())
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "", cleaned)
    return cleaned or "unknown"


def _canonical_timepoint(value: Any) -> str:
    try:
        return str(float(value))
    except (TypeError, ValueError):
        return str(value)


def _embedding_columns(df: pd.DataFrame) -> list[Any]:
    columns = [
        column
        for column in df.columns
        if isinstance(column, (int, np.integer))
        or (isinstance(column, str) and column.isdigit())
    ]
    if not columns:
        raise ValueError("No embedding coordinate columns found for held-out timepoint UMAP.")
    return columns


def _load_annotation_map(path: str | Path | None, annotation_column: str) -> dict[str, str]:
    if path is None or not Path(path).exists():
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if isinstance(values, dict) and annotation_column in values and isinstance(values[annotation_column], dict):
        return {str(key): str(value) for key, value in values[annotation_column].items()}
    return {str(key): str(value) for key, value in values.items()}


def _align_extra_obs(df_cells: pd.DataFrame, adata, columns: list[str]) -> pd.DataFrame:
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    df_cells = df_cells.copy()
    df_cells.index = df_cells.index.astype(str)
    for column in columns:
        if column in obs.columns:
            df_cells[column] = df_cells.index.map(obs[column].to_dict())
    return df_cells


def heldout_timepoint_umap_plots(
    embeddings_dict: dict[str, dict[str, dict[str, Any]]],
    adata,
    annotation_column: str,
    output_dir: str | Path,
    evaluation_config: EvaluationConfig,
    celltype_groups: Mapping[str, set[str] | list[str] | tuple[str, ...]],
    heldout_timepoints: list[str],
    time_column: str = "time",
    sample_column: str = "sample",
    background_alpha: float = 0.12,
    highlighted_alpha: float = 0.9,
) -> dict[str, dict[str, dict[str, Any]]]:
    adata = adata.copy()
    heldout_labels = [_canonical_timepoint(value) for value in heldout_timepoints]
    figure_root = Path(output_dir)
    plotter_class = get_umap_plotter_class()

    for model_name, model_data in embeddings_dict.items():
        for dataset_name, dataset_meta in model_data.items():
            cell_info = dataset_meta.get("df_cells")
            if cell_info is None or "path" not in cell_info:
                continue

            embedding_run_dir = Path(cell_info["path"]).parent
            run_timestamp = embedding_run_dir.name
            figure_run_dir = create_evaluation_run_directory(
                output_dir=figure_root,
                model_name=model_name,
                dataset_name=dataset_name,
                evaluation_name="heldout_timepoint_umap",
                timestamp=run_timestamp,
            )
            derived_umap_dir = embedding_run_dir / "umap" / "heldout_timepoint_umap" / figure_run_dir.name
            derived_umap_dir.mkdir(parents=True, exist_ok=True)

            df_cells = pd.read_parquet(cell_info["path"])
            df_cells.index = df_cells.index.astype(str)
            annotation_map = _load_annotation_map(cell_info.get("annotation_map"), annotation_column)
            df_cells[annotation_column] = df_cells.index.map(lambda idx: annotation_map.get(str(idx), "unknown"))
            df_cells = _align_extra_obs(df_cells, adata, [time_column, sample_column])
            if time_column not in df_cells.columns:
                raise ValueError(f"Missing `{time_column}` in AnnData observations.")

            df_celltypes = None
            if "df_celltypes" in dataset_meta and dataset_meta["df_celltypes"].get("path"):
                ct_info = dataset_meta["df_celltypes"]
                df_celltypes = pd.read_parquet(ct_info["path"])
                df_celltypes.index = df_celltypes.index.astype(str)
                ct_annotation_map = _load_annotation_map(ct_info.get("annotation_map"), annotation_column)
                df_celltypes[annotation_column] = df_celltypes.index.map(
                    lambda idx: ct_annotation_map.get(str(idx), "unknown")
                )

            group_artifacts: dict[str, dict[str, Any]] = {}
            for group_name, group_values in celltype_groups.items():
                group_celltypes = {str(value) for value in group_values}
                df_group = df_cells[df_cells[annotation_column].astype(str).isin(group_celltypes)].copy()
                if df_group.empty:
                    continue

                df_ct_group = None
                if df_celltypes is not None:
                    present_celltypes = set(df_group[annotation_column].astype(str).unique())
                    df_ct_group = df_celltypes[
                        df_celltypes[annotation_column].astype(str).isin(present_celltypes)
                    ].copy()

                df_group["batch"] = "cell"
                frames = [df_group]
                if df_ct_group is not None and not df_ct_group.empty:
                    df_ct_group["batch"] = "celltype"
                    frames.append(df_ct_group)
                combined = pd.concat(frames, axis=0, ignore_index=False, join="outer")
                embedding_columns = _embedding_columns(combined)
                coords = compute_umap(combined.loc[:, embedding_columns].to_numpy(dtype=np.float32), evaluation_config)
                combined[["UMAP1", "UMAP2"]] = coords

                df_cells_umap = combined[combined["batch"] == "cell"].copy()
                df_cells_umap[time_column] = pd.to_numeric(df_cells_umap[time_column], errors="coerce")
                df_cells_umap["_timepoint_label"] = df_cells_umap[time_column].map(_canonical_timepoint)
                heldout_mask = df_cells_umap["_timepoint_label"].isin(heldout_labels)
                df_cells_umap["_is_heldout_timepoint"] = heldout_mask

                centroids = None
                if df_ct_group is not None and not df_ct_group.empty:
                    centroids = (
                        combined.loc[
                            combined["batch"] == "celltype",
                            [annotation_column, "UMAP1", "UMAP2"],
                        ]
                        .rename(columns={annotation_column: "cell_type"})
                        .reset_index(drop=True)
                    )

                group_dir = figure_run_dir / _safe_component(group_name)
                group_dir.mkdir(parents=True, exist_ok=True)
                group_umap_dir = derived_umap_dir / _safe_component(group_name)
                group_umap_dir.mkdir(parents=True, exist_ok=True)
                cells_umap_path = group_umap_dir / "df_cells_umap.parquet"
                df_cells_umap.to_parquet(cells_umap_path, index=True)

                if centroids is not None:
                    centroids_path = group_umap_dir / "df_celltypes_umap.parquet"
                    centroids.to_parquet(centroids_path, index=True)
                else:
                    centroids_path = None

                prefix = f"heldout_timepoints_{_safe_component(group_name)}"
                highlight_kwargs = {
                    "highlight_column": "_is_heldout_timepoint",
                    "highlight_values": [True],
                    "background_alpha": background_alpha,
                    "highlighted_alpha": highlighted_alpha,
                }
                plot_title = f"{group_name}: held-out timepoints {', '.join(heldout_labels)}"
                plotter = plotter_class()
                plotter.annotate_centroids = False
                plotter.plot_cells(
                    df_cells_umap,
                    annotation_column=annotation_column,
                    output_path=group_dir / f"cells_colored_by_annotation_{prefix}.pdf",
                    title=plot_title,
                    **highlight_kwargs,
                )
                plotter.plot_cells(
                    df_cells_umap,
                    time_color_column=time_column,
                    output_path=group_dir / f"cells_colored_by_time_{prefix}.pdf",
                    title=plot_title,
                    **highlight_kwargs,
                )

                if centroids is not None:
                    plotter.annotate_centroids = True
                    labeled_annotation_stem = group_dir / f"cells_colored_by_annotation_with_labels_{prefix}"
                    plotter.plot_cells(
                        df_cells_umap,
                        annotation_column=annotation_column,
                        annotate_centroids_df=centroids,
                        output_path=labeled_annotation_stem.with_suffix(".pdf"),
                        title=plot_title,
                        **highlight_kwargs,
                    )
                    plotter.plot_cells(
                        df_cells_umap,
                        annotation_column=annotation_column,
                        annotate_centroids_df=centroids,
                        output_path=labeled_annotation_stem.with_suffix(".svg"),
                        title=plot_title,
                        **highlight_kwargs,
                    )
                    labeled_time_stem = group_dir / f"cells_colored_by_time_with_labels_{prefix}"
                    plotter.plot_cells(
                        df_cells_umap,
                        time_color_column=time_column,
                        annotate_centroids_df=centroids,
                        output_path=labeled_time_stem.with_suffix(".pdf"),
                        title=plot_title,
                        **highlight_kwargs,
                    )
                    plotter.plot_cells(
                        df_cells_umap,
                        time_color_column=time_column,
                        annotate_centroids_df=centroids,
                        output_path=labeled_time_stem.with_suffix(".svg"),
                        title=plot_title,
                        **highlight_kwargs,
                    )

                group_artifacts[str(group_name)] = {
                    "df_cells_umap": {"path": str(cells_umap_path), "n_points": int(len(df_cells_umap))},
                    "df_celltypes_umap": (
                        {"path": str(centroids_path), "n_points": int(len(centroids))}
                        if centroids_path is not None and centroids is not None
                        else None
                    ),
                    "n_highlighted_cells": int(heldout_mask.sum()),
                    "n_background_cells": int((~heldout_mask).sum()),
                    "cell_types": sorted(group_celltypes),
                }

            dataset_meta["heldout_timepoint_umaps"] = group_artifacts
            write_metadata(
                figure_run_dir,
                {
                    "evaluation_name": "heldout_timepoint_umap",
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "embedding_run_dir": str(embedding_run_dir),
                    "umap_run_dir": str(derived_umap_dir),
                    "heldout_timepoints": heldout_labels,
                    "time_column": time_column,
                    "sample_column": sample_column,
                    "annotation_column": annotation_column,
                    "background_alpha": background_alpha,
                    "highlighted_alpha": highlighted_alpha,
                    "celltype_groups": {
                        str(key): sorted(map(str, values))
                        for key, values in celltype_groups.items()
                    },
                    "artifacts": group_artifacts,
                },
            )

    return embeddings_dict
