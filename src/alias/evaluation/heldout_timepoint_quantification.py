from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.patches import Rectangle
import os
from sklearn.neighbors import NearestNeighbors

from alias.evaluation.embedding import load_dataset_embedding_artifacts
from alias.evaluation.pseudotime import PseudotimeConfig, _load_scvi_module, _scvi_setup_kwargs
from alias.util.artifacts import create_evaluation_run_directory, write_metadata
from alias.util.plots.color_definition import COLORMAPS
from alias.util.plots.pub_style import set_pub_style


def _ensure_lamanno_util_path() -> None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        candidate = parent / "scripts" / "revision1_v1" / "LaManno" / "util"
        if candidate.exists():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return


_ensure_lamanno_util_path()
from lineages import LAMANNO_LINEAGES  # noqa: E402


@dataclass
class HeldoutTimepointNeighborConfig:
    lineage: str = "lineage_4"
    k_neighbors: int = 15
    time_column: str = "time"
    sample_column: str = "sample"
    include_adata_genexp: bool = True
    include_scvi: bool = True
    scvi_layer: str | None = "counts"
    scvi_batch_key: str = "sample"
    scvi_max_epochs: int = 200
    scvi_early_stopping: bool = True
    scvi_early_stopping_patience: int = 10
    scvi_batch_size: int = 256
    plot_neighborhood_heatmaps: bool = True
    neighborhood_heatmap_formats: tuple[str, ...] = ("png",)
    neighborhood_heatmap_max_size: int = 1200
    reuse_cached_representations: bool = True


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
        raise ValueError("No embedding coordinate columns found for held-out timepoint quantification.")
    return columns


def _timepoint_sort_key(label: str) -> tuple[int, float | str]:
    try:
        return (0, float(label))
    except ValueError:
        return (1, label)


def _adjacent_timepoint_map(timepoint_labels: pd.Series) -> dict[str, set[str]]:
    ordered = sorted(set(timepoint_labels.dropna().astype(str)), key=_timepoint_sort_key)
    adjacent: dict[str, set[str]] = {}
    for index, label in enumerate(ordered):
        values: set[str] = set()
        if index > 0:
            values.add(ordered[index - 1])
        if index < len(ordered) - 1:
            values.add(ordered[index + 1])
        adjacent[label] = values
    return adjacent


def _align_lineage_cells(
    df_cells: pd.DataFrame,
    adata,
    annotation_column: str,
    config: HeldoutTimepointNeighborConfig,
) -> tuple[pd.DataFrame, Any]:
    if config.lineage not in LAMANNO_LINEAGES:
        raise ValueError(f"Unknown LaManno lineage: {config.lineage}")

    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    df_cells = df_cells.copy()
    df_cells.index = df_cells.index.astype(str)

    shared = df_cells.index.intersection(obs.index)
    if shared.empty:
        raise ValueError("No shared cells between embeddings and AnnData observations.")

    df_cells = df_cells.loc[shared].copy()
    df_cells[annotation_column] = obs.loc[shared, annotation_column].astype(str)
    df_cells[config.time_column] = pd.to_numeric(obs.loc[shared, config.time_column], errors="coerce")
    if config.sample_column in obs.columns:
        df_cells[config.sample_column] = obs.loc[shared, config.sample_column].astype(str)
    df_cells["_timepoint_label"] = df_cells[config.time_column].map(_canonical_timepoint)

    lineage_celltypes = {str(value) for value in LAMANNO_LINEAGES[config.lineage]}
    df_lineage = df_cells[df_cells[annotation_column].astype(str).isin(lineage_celltypes)].copy()
    if df_lineage.empty:
        raise ValueError(f"No cells found for {config.lineage}.")

    adata_lineage = adata[df_lineage.index.tolist()].copy()
    return df_lineage.loc[adata_lineage.obs_names].copy(), adata_lineage


def _compute_neighbor_metrics(
    embeddings: np.ndarray,
    df_lineage: pd.DataFrame,
    heldout_labels: set[str],
    *,
    representation: str,
    model_name: str,
    dataset_name: str,
    annotation_column: str,
    config: HeldoutTimepointNeighborConfig,
) -> pd.DataFrame:
    if embeddings.shape[0] != len(df_lineage):
        raise ValueError("Embedding matrix row count does not match lineage dataframe.")
    if len(df_lineage) < 2:
        return pd.DataFrame()

    requested_k = int(config.k_neighbors)
    n_neighbors_fit = min(len(df_lineage), requested_k + 1)
    neighbor_model = NearestNeighbors(n_neighbors=n_neighbors_fit, metric="euclidean")
    neighbor_model.fit(embeddings)

    time_labels = df_lineage["_timepoint_label"].astype(str)
    adjacent_by_timepoint = _adjacent_timepoint_map(time_labels)
    heldout_positions = np.flatnonzero(time_labels.isin(heldout_labels).to_numpy())
    rows: list[dict[str, Any]] = []

    for position in heldout_positions:
        distances, indices = neighbor_model.kneighbors(embeddings[position : position + 1])
        neighbor_positions = [idx for idx in indices[0].tolist() if idx != position][:requested_k]
        if not neighbor_positions:
            continue

        cell_id = str(df_lineage.index[position])
        cell_timepoint = str(time_labels.iloc[position])
        adjacent_timepoints = adjacent_by_timepoint.get(cell_timepoint, set())
        neighbor_df = df_lineage.iloc[neighbor_positions]
        neighbor_timepoints = neighbor_df["_timepoint_label"].astype(str)
        neighbor_distances = [
            float(distance)
            for distance, neighbor_idx in zip(distances[0].tolist(), indices[0].tolist())
            if neighbor_idx != position
        ][: len(neighbor_positions)]

        same_time = neighbor_timepoints == cell_timepoint
        adjacent_time = neighbor_timepoints.isin(adjacent_timepoints)
        same_celltype = neighbor_df[annotation_column].astype(str) == str(
            df_lineage.iloc[position][annotation_column]
        )
        numeric_neighbor_time = pd.to_numeric(neighbor_df[config.time_column], errors="coerce")
        numeric_cell_time = float(df_lineage.iloc[position][config.time_column])

        rows.append(
            {
                "model_name": model_name,
                "dataset_name": dataset_name,
                "representation": representation,
                "lineage": config.lineage,
                "cell_id": cell_id,
                "celltype": str(df_lineage.iloc[position][annotation_column]),
                "time": numeric_cell_time,
                "timepoint_label": cell_timepoint,
                "k_requested": requested_k,
                "n_neighbors_used": len(neighbor_positions),
                "fraction_same_timepoint_neighbors": float(same_time.mean()),
                "fraction_adjacent_timepoint_neighbors": float(adjacent_time.mean()),
                "fraction_same_or_adjacent_timepoint_neighbors": float((same_time | adjacent_time).mean()),
                "fraction_same_celltype_neighbors": float(same_celltype.mean()),
                "mean_abs_time_delta": float((numeric_neighbor_time - numeric_cell_time).abs().mean()),
                "mean_neighbor_distance": float(np.mean(neighbor_distances)),
                "neighbor_cell_ids": ";".join(map(str, neighbor_df.index.tolist())),
                "neighbor_timepoints": ";".join(neighbor_timepoints.tolist()),
            }
        )

    return pd.DataFrame(rows)


def _sorted_lineage_cell_ids(
    df_lineage: pd.DataFrame,
    heldout_labels: set[str],
    annotation_column: str,
    config: HeldoutTimepointNeighborConfig,
) -> list[str]:
    sortable = df_lineage.copy()
    sortable["_sort_time"] = pd.to_numeric(sortable[config.time_column], errors="coerce")
    sortable["_sort_celltype"] = sortable[annotation_column].astype(str)
    sortable["_sort_heldout"] = ~sortable["_timepoint_label"].astype(str).isin(heldout_labels)
    sort_columns = ["_sort_time", "_sort_celltype", "_sort_heldout"]
    if config.sample_column in sortable.columns:
        sortable["_sort_sample"] = sortable[config.sample_column].astype(str)
        sort_columns.append("_sort_sample")
    return sortable.sort_values(sort_columns, kind="mergesort").index.astype(str).tolist()


def _rank_weighted_affinity_matrix(
    embeddings: np.ndarray,
    cell_ids: list[str],
    *,
    k_neighbors: int,
    sorted_cell_ids: list[str],
) -> pd.DataFrame:
    if embeddings.shape[0] != len(cell_ids):
        raise ValueError("Embedding matrix row count does not match cell IDs.")
    affinity = pd.DataFrame(0.0, index=cell_ids, columns=cell_ids)
    if len(cell_ids) < 2:
        return affinity.loc[sorted_cell_ids, sorted_cell_ids]

    n_neighbors_fit = min(len(cell_ids), int(k_neighbors) + 1)
    neighbor_model = NearestNeighbors(n_neighbors=n_neighbors_fit, metric="euclidean")
    neighbor_model.fit(embeddings)
    _, indices = neighbor_model.kneighbors(embeddings)

    for query_position, neighbor_positions in enumerate(indices):
        query_id = cell_ids[query_position]
        rank = 1
        for neighbor_position in neighbor_positions:
            if neighbor_position == query_position:
                continue
            if rank > k_neighbors:
                break
            affinity.loc[query_id, cell_ids[neighbor_position]] = 1.0 / np.sqrt(rank)
            rank += 1

    return affinity.loc[sorted_cell_ids, sorted_cell_ids]


def _heldout_runs(sorted_cell_ids: list[str], heldout_cell_ids: set[str]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, cell_id in enumerate(sorted_cell_ids):
        is_heldout = cell_id in heldout_cell_ids
        if is_heldout and start is None:
            start = index
        elif not is_heldout and start is not None:
            runs.append((start, index - start))
            start = None
    if start is not None:
        runs.append((start, len(sorted_cell_ids) - start))
    return runs


def _log(message: str) -> None:
    print(f"[heldout-neighborhood] {message}", flush=True)


def _max_pool_matrix(
    matrix: pd.DataFrame,
    max_size: int,
    *,
    preserve_abs: bool = False,
) -> pd.DataFrame:
    if max_size <= 0 or len(matrix) <= max_size:
        return matrix
    values = matrix.to_numpy()
    n_rows, n_cols = values.shape
    row_edges = np.linspace(0, n_rows, max_size + 1, dtype=int)
    col_edges = np.linspace(0, n_cols, max_size + 1, dtype=int)
    pooled = np.zeros((max_size, max_size), dtype=values.dtype)
    for row_idx in range(max_size):
        row_start, row_end = row_edges[row_idx], row_edges[row_idx + 1]
        for col_idx in range(max_size):
            col_start, col_end = col_edges[col_idx], col_edges[col_idx + 1]
            block = values[row_start:row_end, col_start:col_end]
            if not block.size:
                continue
            if preserve_abs:
                flat = block.ravel()
                pooled[row_idx, col_idx] = flat[np.nanargmax(np.abs(flat))]
            else:
                pooled[row_idx, col_idx] = np.nanmax(block)
    return pd.DataFrame(pooled)


def _scale_runs_for_pooled_matrix(
    runs: list[tuple[int, int]],
    *,
    original_size: int,
    plotted_size: int,
) -> list[tuple[float, float]]:
    if original_size == plotted_size:
        return [(float(start), float(length)) for start, length in runs]
    scale = plotted_size / original_size
    return [(start * scale, max(1.0, length * scale)) for start, length in runs]


def _plot_rank_weighted_heatmap(
    matrix: pd.DataFrame,
    *,
    output_stem: Path,
    title: str,
    heldout_cell_ids: set[str],
    formats: tuple[str, ...],
    max_size: int,
    difference: bool = False,
) -> None:
    set_pub_style()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    original_size = len(matrix)
    plot_matrix = _max_pool_matrix(matrix, max_size=max_size, preserve_abs=difference)
    plotted_size = len(plot_matrix)
    if plotted_size != original_size:
        _log(f"Downsampled heatmap {output_stem.name}: {original_size}x{original_size} -> {plotted_size}x{plotted_size}")
    size = max(4.0, min(10.0, plotted_size * 0.16)) * 1.5
    fig, ax = plt.subplots(figsize=(size, size))
    cmap = COLORMAPS["Heatmap: Teal–White–Red"] if difference else COLORMAPS["CM: Teal–White"]
    vmax = float(np.nanmax(np.abs(plot_matrix.to_numpy()))) if difference else 1.0
    vmin = -vmax if difference else 0.0
    sns.heatmap(
        plot_matrix,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Rank-weighted affinity", "shrink": 0.6},
    )
    colorbar = ax.collections[0].colorbar
    if colorbar is not None:
        colorbar.outline.set_visible(True)
        colorbar.outline.set_edgecolor("black")
        colorbar.outline.set_linewidth(1.2)
        colorbar.ax.tick_params(labelsize=34, width=1.0, length=4, color="black")
        colorbar.ax.yaxis.label.set_size(38)
        colorbar.ax.yaxis.label.set_color("black")

    sorted_cell_ids = matrix.index.astype(str).tolist()
    heldout_runs = _scale_runs_for_pooled_matrix(
        _heldout_runs(sorted_cell_ids, heldout_cell_ids),
        original_size=original_size,
        plotted_size=plotted_size,
    )
    for start, length in heldout_runs:
        ax.add_patch(Rectangle((0, start), plotted_size, length, fill=False, edgecolor="black", linewidth=1.2))
        ax.add_patch(Rectangle((start, 0), length, plotted_size, fill=False, edgecolor="black", linewidth=1.2))

    ax.set_xlabel("Neighbor cells", fontsize=38)
    ax.set_ylabel("Query cells", fontsize=38)
    ax.set_title(title, fontsize=38)
    fig.tight_layout()
    mpl.rcParams["svg.fonttype"] = "none"
    for fmt in formats:
        suffix = f".{fmt.lstrip('.')}"
        final_path = output_stem.with_suffix(suffix)
        tmp_path = output_stem.with_name(f"{output_stem.name}.tmp{suffix}")
        fig.savefig(tmp_path, bbox_inches="tight", dpi=300)
        os.replace(tmp_path, final_path)
    plt.close(fig)


def _neighbor_overlap_table(
    first: pd.DataFrame,
    second: pd.DataFrame,
    heldout_cell_ids: set[str],
    *,
    heldout_only: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cell_id in first.index.astype(str):
        is_heldout = cell_id in heldout_cell_ids
        if heldout_only and not is_heldout:
            continue
        first_weights = first.loc[cell_id]
        second_weights = second.loc[cell_id]
        first_neighbors = set(first_weights[first_weights > 0].index.astype(str))
        second_neighbors = set(second_weights[second_weights > 0].index.astype(str))
        union = first_neighbors.union(second_neighbors)
        intersection = first_neighbors.intersection(second_neighbors)
        max_weights = np.maximum(first_weights.to_numpy(), second_weights.to_numpy()).sum()
        min_weights = np.minimum(first_weights.to_numpy(), second_weights.to_numpy()).sum()
        rows.append(
            {
                "cell_id": cell_id,
                "is_heldout_cell": bool(is_heldout),
                "top_k_jaccard": float(len(intersection) / len(union)) if union else np.nan,
                "weighted_overlap": float(min_weights / max_weights) if max_weights > 0 else np.nan,
                "n_first_neighbors": int(len(first_neighbors)),
                "n_second_neighbors": int(len(second_neighbors)),
                "n_shared_neighbors": int(len(intersection)),
            }
        )
    return pd.DataFrame(rows)


def _neighbor_overlap_summary(
    overlap: pd.DataFrame,
    df_lineage: pd.DataFrame,
    annotation_column: str,
) -> pd.DataFrame:
    if overlap.empty:
        return pd.DataFrame(
            columns=[
                "timepoint_label",
                "celltype",
                "mean_weighted_overlap",
                "median_weighted_overlap",
                "mean_top_k_jaccard",
                "median_top_k_jaccard",
                "n_heldout_cells",
            ]
        )

    metadata = df_lineage[["_timepoint_label", annotation_column]].copy()
    metadata.index = metadata.index.astype(str)
    enriched = overlap.copy()
    enriched["cell_id"] = enriched["cell_id"].astype(str)
    enriched["timepoint_label"] = enriched["cell_id"].map(metadata["_timepoint_label"].astype(str))
    enriched["celltype"] = enriched["cell_id"].map(metadata[annotation_column].astype(str))
    if "is_heldout_cell" not in enriched.columns:
        enriched["is_heldout_cell"] = False
    summary = (
        enriched.groupby(["timepoint_label", "celltype"], dropna=False, as_index=False)
        .agg(
            mean_weighted_overlap=("weighted_overlap", "mean"),
            median_weighted_overlap=("weighted_overlap", "median"),
            mean_top_k_jaccard=("top_k_jaccard", "mean"),
            median_top_k_jaccard=("top_k_jaccard", "median"),
            n_cells=("cell_id", "size"),
            n_heldout_cells=("is_heldout_cell", "sum"),
        )
        .sort_values(
            ["timepoint_label", "celltype"],
            key=lambda values: values.map(_timepoint_sort_key) if values.name == "timepoint_label" else values,
            kind="mergesort",
        )
    )
    return summary.reset_index(drop=True)


def _overlap_timepoint_barplot_dataframe(overlap_summary: pd.DataFrame) -> pd.DataFrame:
    if overlap_summary.empty:
        return pd.DataFrame(
            columns=[
                "timepoint",
                "mean_weighted_overlap",
                "mean_top_k_jaccard",
                "n_heldout_cells",
            ]
        )

    plot_df = (
        overlap_summary.groupby("timepoint_label", as_index=False, sort=False)
        .apply(
            lambda frame: pd.Series(
                {
                    "mean_weighted_overlap": np.average(
                        frame["mean_weighted_overlap"].astype(float),
                        weights=frame["n_cells"].astype(float),
                    ),
                    "mean_top_k_jaccard": np.average(
                        frame["mean_top_k_jaccard"].astype(float),
                        weights=frame["n_cells"].astype(float),
                    ),
                    "n_cells": int(frame["n_cells"].sum()),
                    "n_heldout_cells": int(frame["n_heldout_cells"].sum()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .rename(columns={"timepoint_label": "timepoint"})
    )
    plot_df["timepoint"] = plot_df["timepoint"].astype(str)
    plot_df = plot_df[
        plot_df["timepoint"].notna()
        & ~plot_df["timepoint"].isin({"", "nan", "None", "<NA>"})
        & (plot_df["n_cells"].astype(int) > 0)
    ]
    plot_df = plot_df.sort_values("timepoint", key=lambda values: values.map(_timepoint_sort_key), kind="mergesort")
    return plot_df.reset_index(drop=True)


def _plot_neighbor_overlap_metric_timepoint_barplot(
    overlap_summary: pd.DataFrame,
    *,
    output_stem: Path,
    heldout_labels: set[str],
    metric_column: str,
    ylabel: str,
    title: str,
) -> None:
    plot_df = _overlap_timepoint_barplot_dataframe(overlap_summary)
    if plot_df.empty:
        return

    set_pub_style()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    timepoints = plot_df["timepoint"].astype(str).tolist()
    palette = _magma_timepoint_palette(timepoints)
    x_positions = np.arange(len(timepoints), dtype=float) * 0.48
    bar_width = 0.16
    fig_width = max(1.0, (1.0 + len(timepoints) * 0.21) * (2.0 / 3.0))
    fig, ax = plt.subplots(figsize=(fig_width, 2.4))
    bars = ax.bar(
        x_positions,
        plot_df[metric_column].astype(float),
        width=bar_width,
        color=[palette[timepoint] for timepoint in timepoints],
        edgecolor="0.15",
        linewidth=0.35,
    )
    for bar, timepoint in zip(bars, timepoints):
        bar.set_alpha(1.0 if timepoint in heldout_labels else 0.35)

    ax.set_xlabel("Timepoint", fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=7.5)
    ax.set_title(title, fontsize=7.5)
    values = plot_df[metric_column].astype(float)
    upper = min(1.0, max(0.05, float(np.nanmax(values)) * 1.15))
    ax.set_ylim(0.0, upper)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(timepoints, rotation=45, ha="right", fontsize=7.5)
    ax.tick_params(axis="x", which="major", bottom=True, length=3, width=0.7, color="0.2")
    ax.tick_params(axis="y", labelsize=7.5)
    if len(x_positions):
        ax.set_xlim(x_positions.min() - 0.32, x_positions.max() + 0.32)
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    fig.tight_layout()

    for suffix in (".png", ".pdf"):
        final_path = output_stem.with_suffix(suffix)
        tmp_path = output_stem.with_name(f"{output_stem.name}.tmp{suffix}")
        fig.savefig(tmp_path, bbox_inches="tight", dpi=300)
        os.replace(tmp_path, final_path)
    plt.close(fig)


def _timepoint_affinity_difference_summary(
    alias_matrix: pd.DataFrame,
    scvi_matrix: pd.DataFrame,
    df_lineage: pd.DataFrame,
    heldout_labels: set[str],
) -> pd.DataFrame:
    diff = alias_matrix - scvi_matrix
    timepoints = df_lineage["_timepoint_label"].astype(str)
    query_timepoints = pd.Series(diff.index.astype(str), index=diff.index).map(timepoints)
    neighbor_timepoints = pd.Series(diff.columns.astype(str), index=diff.columns).map(timepoints)
    query_counts = query_timepoints.value_counts().astype(int).to_dict()
    neighbor_counts = neighbor_timepoints.value_counts().astype(int).to_dict()
    rows: list[dict[str, Any]] = []

    ordered_query_timepoints = sorted(
        {
            str(timepoint)
            for timepoint in query_timepoints.dropna()
            if str(timepoint) not in {"", "nan", "None", "<NA>"}
            and int(query_counts.get(timepoint, 0)) > 0
        },
        key=_timepoint_sort_key,
    )
    query_timepoint_set = set(ordered_query_timepoints)
    ordered_neighbor_timepoints = sorted(
        {
            str(timepoint)
            for timepoint in neighbor_timepoints.dropna()
            if str(timepoint) in query_timepoint_set
        },
        key=_timepoint_sort_key,
    )
    for query_timepoint in ordered_query_timepoints:
        query_mask = query_timepoints == query_timepoint
        for neighbor_timepoint in ordered_neighbor_timepoints:
            neighbor_mask = neighbor_timepoints == neighbor_timepoint
            values = diff.loc[query_mask.to_numpy(), neighbor_mask.to_numpy()].to_numpy(dtype=float)
            flat_values = values.ravel()
            nonzero_values = flat_values[flat_values != 0]
            is_heldout = query_timepoint in heldout_labels or neighbor_timepoint in heldout_labels
            rows.append(
                {
                    "query_timepoint": str(query_timepoint),
                    "neighbor_timepoint": str(neighbor_timepoint),
                    "mean_alias_minus_scvi_rank_weighted_affinity": float(np.nanmean(flat_values)),
                    "median_alias_minus_scvi_rank_weighted_affinity": float(np.nanmedian(flat_values)),
                    "sum_alias_minus_scvi_rank_weighted_affinity": float(np.nansum(flat_values)),
                    "mean_nonzero_alias_minus_scvi_rank_weighted_affinity": (
                        float(np.nanmean(nonzero_values)) if len(nonzero_values) else 0.0
                    ),
                    "n_query_cells": int(query_counts.get(query_timepoint, 0)),
                    "n_neighbor_cells": int(neighbor_counts.get(neighbor_timepoint, 0)),
                    "n_cell_pairs": int(flat_values.size),
                    "n_nonzero_cell_pairs": int(len(nonzero_values)),
                    "is_heldout_timepoint": bool(is_heldout),
                    "query_is_heldout_timepoint": bool(query_timepoint in heldout_labels),
                    "neighbor_is_heldout_timepoint": bool(neighbor_timepoint in heldout_labels),
                }
            )

    return pd.DataFrame(rows)


def _magma_timepoint_palette(timepoints: list[str]) -> dict[str, tuple[float, float, float, float]]:
    ordered_timepoints = sorted([str(value) for value in timepoints], key=_timepoint_sort_key)
    if not ordered_timepoints:
        return {}
    cmap = COLORMAPS["magma"]
    if len(ordered_timepoints) == 1:
        values = [0.68]
    else:
        values = np.linspace(0.18, 0.88, len(ordered_timepoints))
    return {
        timepoint: tuple(cmap(float(value)))
        for timepoint, value in zip(ordered_timepoints, values)
    }


def _timepoint_barplot_dataframe(summary: pd.DataFrame) -> pd.DataFrame:
    valid_summary = summary.copy()
    valid_summary["query_timepoint"] = valid_summary["query_timepoint"].astype(str)
    valid_summary = valid_summary[
        valid_summary["query_timepoint"].notna()
        & ~valid_summary["query_timepoint"].isin({"", "nan", "None", "<NA>"})
        & (valid_summary["n_query_cells"].astype(int) > 0)
    ]
    if valid_summary.empty:
        return pd.DataFrame(
            columns=[
                "timepoint",
                "mean_alias_minus_scvi_rank_weighted_affinity",
                "n_cell_pairs",
                "n_query_cells",
            ]
        )

    plot_df = (
        valid_summary.groupby("query_timepoint", as_index=False, sort=False)
        .agg(
            sum_alias_minus_scvi_rank_weighted_affinity=(
                "sum_alias_minus_scvi_rank_weighted_affinity",
                "sum",
            ),
            n_cell_pairs=("n_cell_pairs", "sum"),
            n_query_cells=("n_query_cells", "max"),
        )
        .rename(columns={"query_timepoint": "timepoint"})
    )
    plot_df["mean_alias_minus_scvi_rank_weighted_affinity"] = (
        plot_df["sum_alias_minus_scvi_rank_weighted_affinity"] / plot_df["n_cell_pairs"]
    )
    plot_df = plot_df.sort_values("timepoint", key=lambda values: values.map(_timepoint_sort_key), kind="mergesort")
    return plot_df[
        [
            "timepoint",
            "mean_alias_minus_scvi_rank_weighted_affinity",
            "n_cell_pairs",
            "n_query_cells",
        ]
    ].reset_index(drop=True)


def _plot_timepoint_affinity_difference_barplot(
    summary: pd.DataFrame,
    *,
    output_path: Path,
    heldout_labels: set[str],
) -> None:
    if summary.empty:
        return

    set_pub_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = _timepoint_barplot_dataframe(summary)
    if plot_df.empty:
        return
    timepoints = plot_df["timepoint"].astype(str).tolist()
    palette = _magma_timepoint_palette(timepoints)

    x_positions = np.arange(len(timepoints), dtype=float) * 0.62
    fig_width = max(3.0, 1.0 + len(timepoints) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_width, 2.4))
    bars = ax.bar(
        x_positions,
        plot_df["mean_alias_minus_scvi_rank_weighted_affinity"].astype(float),
        width=0.22,
        color=[palette[timepoint] for timepoint in timepoints],
        edgecolor="0.15",
        linewidth=0.35,
    )
    for bar, timepoint in zip(bars, timepoints):
        bar.set_alpha(1.0 if timepoint in heldout_labels else 0.35)

    ax.axhline(0.0, color="0.2", linewidth=0.7)
    ax.set_xlabel("Query timepoint", fontsize=7.5)
    ax.set_ylabel("Mean ALIAS - scVI affinity", fontsize=7.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(timepoints, rotation=30, ha="right", fontsize=7.5)
    ax.tick_params(axis="x", which="major", bottom=True, length=3, width=0.7, color="0.2")
    ax.tick_params(axis="y", labelsize=7.5)
    if len(x_positions):
        ax.set_xlim(x_positions.min() - 0.32, x_positions.max() + 0.32)
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    fig.tight_layout()

    tmp_path = output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    fig.savefig(tmp_path, bbox_inches="tight", dpi=300)
    os.replace(tmp_path, output_path)
    plt.close(fig)


def _write_neighborhood_visualizations(
    representations: dict[str, np.ndarray],
    df_lineage: pd.DataFrame,
    heldout_labels: set[str],
    lineage_dir: Path,
    annotation_column: str,
    config: HeldoutTimepointNeighborConfig,
) -> dict[str, str]:
    _log("Preparing rank-weighted neighborhood heatmaps...")
    sorted_cell_ids = _sorted_lineage_cell_ids(df_lineage, heldout_labels, annotation_column, config)
    heldout_cell_ids = set(
        df_lineage.index[df_lineage["_timepoint_label"].astype(str).isin(heldout_labels)].astype(str)
    )
    cell_ids = df_lineage.index.astype(str).tolist()
    matrices = {
        name: _rank_weighted_affinity_matrix(
            embeddings,
            cell_ids,
            k_neighbors=config.k_neighbors,
            sorted_cell_ids=sorted_cell_ids,
        )
        for name, embeddings in representations.items()
        if name in {"alias", "scvi"}
    }
    artifacts: dict[str, str] = {}
    for name, matrix in matrices.items():
        stem = lineage_dir / f"{name}_rank_weighted_neighborhood_heatmap"
        _log(f"Rendering {name} neighborhood heatmap ({len(matrix)} cells)...")
        _plot_rank_weighted_heatmap(
            matrix,
            output_stem=stem,
            title=f"{name} rank-weighted neighborhoods",
            heldout_cell_ids=heldout_cell_ids,
            formats=config.neighborhood_heatmap_formats,
            max_size=config.neighborhood_heatmap_max_size,
        )
        for fmt in config.neighborhood_heatmap_formats:
            suffix = f".{fmt.lstrip('.')}"
            artifacts[f"{name}_heatmap_{fmt.lstrip('.')}"] = str(stem.with_suffix(suffix))

    if {"alias", "scvi"} <= set(matrices):
        diff = matrices["alias"] - matrices["scvi"]
        diff_stem = lineage_dir / "alias_minus_scvi_rank_weighted_neighborhood_heatmap"
        _log("Rendering alias-minus-scVI neighborhood difference heatmap...")
        _plot_rank_weighted_heatmap(
            diff,
            output_stem=diff_stem,
            title="alias minus scvi rank-weighted neighborhoods",
            heldout_cell_ids=heldout_cell_ids,
            formats=config.neighborhood_heatmap_formats,
            max_size=config.neighborhood_heatmap_max_size,
            difference=True,
        )
        overlap = _neighbor_overlap_table(matrices["alias"], matrices["scvi"], heldout_cell_ids)
        overlap_path = lineage_dir / "heldout_neighbor_overlap.csv"
        overlap.to_csv(overlap_path, index=False)
        _log(f"Saved held-out neighbor overlap table to {overlap_path}")
        all_overlap = _neighbor_overlap_table(
            matrices["alias"],
            matrices["scvi"],
            heldout_cell_ids,
            heldout_only=False,
        )
        all_overlap_metadata = df_lineage[["_timepoint_label", annotation_column]].copy()
        all_overlap_metadata.index = all_overlap_metadata.index.astype(str)
        all_overlap["timepoint_label"] = all_overlap["cell_id"].astype(str).map(
            all_overlap_metadata["_timepoint_label"].astype(str)
        )
        all_overlap["celltype"] = all_overlap["cell_id"].astype(str).map(
            all_overlap_metadata[annotation_column].astype(str)
        )
        all_overlap_path = lineage_dir / "neighbor_overlap_all_cells.csv"
        all_overlap.to_csv(all_overlap_path, index=False)
        _log(f"Saved all-cell neighbor overlap table to {all_overlap_path}")
        overlap_summary = _neighbor_overlap_summary(all_overlap, df_lineage, annotation_column)
        overlap_summary_path = lineage_dir / "heldout_neighbor_overlap_summary.csv"
        weighted_overlap_barplot_stem = lineage_dir / "neighbor_weighted_overlap_timepoint_barplot"
        jaccard_barplot_stem = lineage_dir / "neighbor_top_k_jaccard_timepoint_barplot"
        overlap_summary.to_csv(overlap_summary_path, index=False)
        _log(f"Saved held-out neighbor overlap summary to {overlap_summary_path}")
        _plot_neighbor_overlap_metric_timepoint_barplot(
            overlap_summary,
            output_stem=weighted_overlap_barplot_stem,
            heldout_labels=heldout_labels,
            metric_column="mean_weighted_overlap",
            ylabel="Weighted overlap",
            title="Weighted neighbor overlap",
        )
        _plot_neighbor_overlap_metric_timepoint_barplot(
            overlap_summary,
            output_stem=jaccard_barplot_stem,
            heldout_labels=heldout_labels,
            metric_column="mean_top_k_jaccard",
            ylabel="Top-k Jaccard",
            title="Top-k neighbor Jaccard",
        )
        _log(
            "Saved neighbor overlap bar plots to "
            f"{weighted_overlap_barplot_stem.with_suffix('.png')}, "
            f"{weighted_overlap_barplot_stem.with_suffix('.pdf')}, "
            f"{jaccard_barplot_stem.with_suffix('.png')}, and "
            f"{jaccard_barplot_stem.with_suffix('.pdf')}"
        )
        timepoint_summary = _timepoint_affinity_difference_summary(
            matrices["alias"],
            matrices["scvi"],
            df_lineage,
            heldout_labels,
        )
        timepoint_summary_path = lineage_dir / "alias_minus_scvi_rank_weighted_timepoint_summary.csv"
        timepoint_barplot_path = lineage_dir / "alias_minus_scvi_rank_weighted_timepoint_barplot.png"
        timepoint_summary.to_csv(timepoint_summary_path, index=False)
        _log(f"Saved alias-minus-scVI timepoint summary to {timepoint_summary_path}")
        _plot_timepoint_affinity_difference_barplot(
            timepoint_summary,
            output_path=timepoint_barplot_path,
            heldout_labels=heldout_labels,
        )
        _log(f"Saved alias-minus-scVI timepoint bar plot to {timepoint_barplot_path}")
        for fmt in config.neighborhood_heatmap_formats:
            suffix = f".{fmt.lstrip('.')}"
            artifacts[f"alias_minus_scvi_heatmap_{fmt.lstrip('.')}"] = str(diff_stem.with_suffix(suffix))
        artifacts["heldout_neighbor_overlap"] = str(overlap_path)
        artifacts["neighbor_overlap_all_cells"] = str(all_overlap_path)
        artifacts["heldout_neighbor_overlap_summary"] = str(overlap_summary_path)
        artifacts["neighbor_weighted_overlap_timepoint_barplot_png"] = str(
            weighted_overlap_barplot_stem.with_suffix(".png")
        )
        artifacts["neighbor_weighted_overlap_timepoint_barplot_pdf"] = str(
            weighted_overlap_barplot_stem.with_suffix(".pdf")
        )
        artifacts["neighbor_top_k_jaccard_timepoint_barplot_png"] = str(jaccard_barplot_stem.with_suffix(".png"))
        artifacts["neighbor_top_k_jaccard_timepoint_barplot_pdf"] = str(jaccard_barplot_stem.with_suffix(".pdf"))
        artifacts["alias_minus_scvi_timepoint_summary"] = str(timepoint_summary_path)
        artifacts["alias_minus_scvi_timepoint_barplot_png"] = str(timepoint_barplot_path)

    return artifacts


def _summary_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    group_columns = ["model_name", "dataset_name", "representation", "lineage", "timepoint_label", "celltype"]
    value_columns = [
        "fraction_same_timepoint_neighbors",
        "fraction_adjacent_timepoint_neighbors",
        "fraction_same_or_adjacent_timepoint_neighbors",
        "fraction_same_celltype_neighbors",
        "mean_abs_time_delta",
        "mean_neighbor_distance",
    ]
    summary = metrics.groupby(group_columns, dropna=False)[value_columns].agg(["mean", "median", "std"])
    summary.columns = ["_".join(column).rstrip("_") for column in summary.columns.to_flat_index()]
    summary["n_heldout_cells"] = metrics.groupby(group_columns, dropna=False).size()
    return summary.reset_index()


def _pseudotime_scvi_config(config: HeldoutTimepointNeighborConfig) -> PseudotimeConfig:
    return PseudotimeConfig(
        lineage=config.lineage,
        cell_origin="",
        scvi_layer=config.scvi_layer,
        scvi_batch_key=config.scvi_batch_key,
        scvi_max_epochs=config.scvi_max_epochs,
        scvi_early_stopping=config.scvi_early_stopping,
        scvi_early_stopping_patience=config.scvi_early_stopping_patience,
        scvi_batch_size=config.scvi_batch_size,
    )


def _run_scvi_latent(adata_lineage, config: HeldoutTimepointNeighborConfig) -> tuple[np.ndarray, dict[str, Any]]:
    scvi_module = _load_scvi_module()
    pseudotime_config = _pseudotime_scvi_config(config)
    setup_kwargs = _scvi_setup_kwargs(adata_lineage, pseudotime_config)
    scvi_module.model.SCVI.setup_anndata(adata_lineage, **setup_kwargs)
    model = scvi_module.model.SCVI(adata_lineage)
    train_kwargs = {
        "max_epochs": config.scvi_max_epochs,
        "early_stopping": config.scvi_early_stopping,
        "early_stopping_patience": config.scvi_early_stopping_patience,
        "batch_size": config.scvi_batch_size,
    }
    model.train(**train_kwargs)
    latent = np.asarray(model.get_latent_representation(), dtype=np.float32)
    adata_lineage.obsm["X_scVI"] = latent
    return latent, {"setup_kwargs": setup_kwargs, "train_kwargs": train_kwargs, "reused": False}


def _run_adata_genexp_pca(adata_lineage) -> tuple[np.ndarray, dict[str, Any]]:
    pca_kwargs = {"n_comps": 50, "svd_solver": "arpack"}
    sc.tl.pca(adata_lineage, **pca_kwargs)
    if "X_pca" not in adata_lineage.obsm:
        raise ValueError("Scanpy PCA did not populate adata.obsm['X_pca'].")
    return np.asarray(adata_lineage.obsm["X_pca"], dtype=np.float32), {"pca_kwargs": pca_kwargs, "reused": False}


def _read_cached_matrix(path: Path, expected_index: pd.Index) -> np.ndarray | None:
    if not path.exists():
        return None
    cached = pd.read_csv(path, index_col=0)
    cached.index = cached.index.astype(str)
    expected = expected_index.astype(str)
    if cached.index.tolist() != expected.tolist():
        return None
    return cached.to_numpy(dtype=np.float32)


def _candidate_cached_matrix_paths(lineage_dir: Path, filename: str) -> list[Path]:
    candidates = [lineage_dir / filename]
    run_parent = lineage_dir.parent.parent
    if run_parent.exists():
        sibling_dirs = sorted(
            (path for path in run_parent.iterdir() if path.is_dir() and path != lineage_dir.parent),
            reverse=True,
        )
        candidates.extend(sibling / lineage_dir.name / filename for sibling in sibling_dirs)
    return candidates


def _read_first_cached_matrix(paths: list[Path], expected_index: pd.Index) -> tuple[np.ndarray, Path] | None:
    for path in paths:
        cached = _read_cached_matrix(path, expected_index)
        if cached is not None:
            return cached, path
    return None


def _load_or_run_adata_genexp_pca(
    adata_lineage,
    df_lineage: pd.DataFrame,
    output_path: Path,
    config: HeldoutTimepointNeighborConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    if config.reuse_cached_representations:
        cached = _read_first_cached_matrix(
            _candidate_cached_matrix_paths(output_path.parent, output_path.name),
            df_lineage.index,
        )
        if cached is not None:
            matrix, source_path = cached
            _log(f"Reusing AnnData expression PCA from {source_path}")
            pd.DataFrame(matrix, index=df_lineage.index).to_csv(output_path)
            return matrix, {
                "pca_kwargs": {"n_comps": 50, "svd_solver": "arpack"},
                "reused": True,
                "source_path": str(source_path),
            }
    _log("Computing AnnData expression PCA...")
    matrix, metadata = _run_adata_genexp_pca(adata_lineage)
    pd.DataFrame(matrix, index=df_lineage.index).to_csv(output_path)
    _log(f"Saved AnnData expression PCA to {output_path}")
    return matrix, metadata


def _load_or_run_scvi_latent(
    adata_lineage,
    df_lineage: pd.DataFrame,
    output_path: Path,
    config: HeldoutTimepointNeighborConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    if config.reuse_cached_representations:
        cached = _read_first_cached_matrix(
            _candidate_cached_matrix_paths(output_path.parent, output_path.name),
            df_lineage.index,
        )
        if cached is not None:
            matrix, source_path = cached
            _log(f"Reusing scVI latent representation from {source_path}")
            pd.DataFrame(matrix, index=df_lineage.index).to_csv(output_path)
            return matrix, {
                "setup_kwargs": None,
                "train_kwargs": None,
                "reused": True,
                "source_path": str(source_path),
            }
    _log("Training scVI latent representation...")
    matrix, metadata = _run_scvi_latent(adata_lineage, config)
    pd.DataFrame(matrix, index=df_lineage.index).to_csv(output_path)
    _log(f"Saved scVI latent representation to {output_path}")
    return matrix, metadata


def heldout_timepoint_neighbor_enrichment(
    embeddings_dict: dict[str, dict[str, dict[str, Any]]],
    eval_data: dict[str, Any],
    subfolder_fig_dir: str | Path,
    annotation_column: str,
    heldout_timepoints: list[str],
    config: HeldoutTimepointNeighborConfig | None = None,
) -> pd.DataFrame:
    config = config or HeldoutTimepointNeighborConfig()
    if "adata" not in eval_data or eval_data["adata"] is None:
        raise ValueError("Held-out timepoint neighbor enrichment requires `eval_data['adata']`.")

    adata = eval_data["adata"].copy()
    heldout_labels = {_canonical_timepoint(value) for value in heldout_timepoints}
    figure_root = Path(subfolder_fig_dir)
    all_metrics: list[pd.DataFrame] = []

    for model_name, model_data in embeddings_dict.items():
        for dataset_name, dataset_meta in model_data.items():
            _log(f"Starting neighborhood assessment for model={model_name}, dataset={dataset_name}")
            loaded_dataset = load_dataset_embedding_artifacts(dataset_meta, annotation_column=annotation_column)
            artifacts = loaded_dataset["artifacts"]
            if "df_cells" not in artifacts:
                _log(f"Skipping {dataset_name}: no df_cells artifact found.")
                continue

            run_timestamp = loaded_dataset["run_timestamp"] or "unknown_embedding_run"
            run_dir = create_evaluation_run_directory(
                output_dir=figure_root,
                model_name=model_name,
                dataset_name=dataset_name,
                evaluation_name="heldout_timepoint_neighbor_enrichment",
                timestamp=run_timestamp,
            )
            lineage_dir = run_dir / config.lineage
            lineage_dir.mkdir(parents=True, exist_ok=True)
            _log(f"Writing neighborhood assessment outputs to {lineage_dir}")

            df_cells = artifacts["df_cells"]["dataframe"].copy()
            df_lineage, adata_lineage = _align_lineage_cells(
                df_cells,
                adata,
                annotation_column,
                config,
            )
            n_heldout = int(df_lineage["_timepoint_label"].isin(heldout_labels).sum())
            _log(f"Aligned {len(df_lineage)} {config.lineage} cells; held-out cells: {n_heldout}")
            embedding_columns = _embedding_columns(df_lineage)
            alias_embeddings = df_lineage.loc[:, embedding_columns].to_numpy(dtype=np.float32)
            _log(f"Computing ALIAS neighbor metrics with k={config.k_neighbors}...")
            alias_metrics = _compute_neighbor_metrics(
                alias_embeddings,
                df_lineage,
                heldout_labels,
                representation="alias",
                model_name=model_name,
                dataset_name=dataset_name,
                annotation_column=annotation_column,
                config=config,
            )

            metrics = [alias_metrics]
            representation_embeddings: dict[str, np.ndarray] = {"alias": alias_embeddings}
            adata_genexp_metadata = None
            if config.include_adata_genexp:
                adata_genexp_pca, adata_genexp_metadata = _load_or_run_adata_genexp_pca(
                    adata_lineage.copy(),
                    df_lineage,
                    lineage_dir / "adata_genexp_pca.csv",
                    config,
                )
                representation_embeddings["adata_genexp"] = adata_genexp_pca
                _log("Computing AnnData expression PCA neighbor metrics...")
                metrics.append(
                    _compute_neighbor_metrics(
                        adata_genexp_pca,
                        df_lineage,
                        heldout_labels,
                        representation="adata_genexp",
                        model_name=model_name,
                        dataset_name=dataset_name,
                        annotation_column=annotation_column,
                        config=config,
                    )
                )

            scvi_metadata = None
            if config.include_scvi:
                scvi_latent, scvi_metadata = _load_or_run_scvi_latent(
                    adata_lineage,
                    df_lineage,
                    lineage_dir / "scvi_latent.csv",
                    config,
                )
                representation_embeddings["scvi"] = scvi_latent
                _log("Computing scVI neighbor metrics...")
                metrics.append(
                    _compute_neighbor_metrics(
                        scvi_latent,
                        df_lineage,
                        heldout_labels,
                        representation="scvi",
                        model_name=model_name,
                        dataset_name=dataset_name,
                        annotation_column=annotation_column,
                        config=config,
                    )
                )

            metrics_df = pd.concat(metrics, ignore_index=True)
            summary_df = _summary_table(metrics_df)
            metrics_df.to_csv(lineage_dir / "neighbor_metrics.csv", index=False)
            summary_df.to_csv(lineage_dir / "summary_by_timepoint_celltype.csv", index=False)
            _log(f"Saved neighbor metrics and summary for {len(metrics_df)} held-out cell rows.")
            neighborhood_plot_artifacts = {}
            if config.plot_neighborhood_heatmaps:
                neighborhood_plot_artifacts = _write_neighborhood_visualizations(
                    representation_embeddings,
                    df_lineage,
                    heldout_labels,
                    lineage_dir,
                    annotation_column,
                    config,
                )
            _log("Writing neighborhood assessment metadata...")
            write_metadata(
                run_dir,
                {
                    "evaluation_name": "heldout_timepoint_neighbor_enrichment",
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "embedding_run_timestamp": run_timestamp,
                    "annotation_column": annotation_column,
                    "heldout_timepoints": sorted(heldout_labels, key=_timepoint_sort_key),
                    "config": asdict(config),
                    "n_lineage_cells": int(len(df_lineage)),
                    "n_heldout_cells": int(df_lineage["_timepoint_label"].isin(heldout_labels).sum()),
                    "artifacts": {
                        "neighbor_metrics": str(lineage_dir / "neighbor_metrics.csv"),
                        "summary_by_timepoint_celltype": str(
                            lineage_dir / "summary_by_timepoint_celltype.csv"
                        ),
                        "adata_genexp_pca": (
                            str(lineage_dir / "adata_genexp_pca.csv")
                            if config.include_adata_genexp
                            else None
                        ),
                        "scvi_latent": str(lineage_dir / "scvi_latent.csv") if config.include_scvi else None,
                        "neighborhood_plots": neighborhood_plot_artifacts,
                    },
                    "adata_genexp": adata_genexp_metadata,
                    "scvi": scvi_metadata,
                },
            )
            all_metrics.append(metrics_df)
            _log(f"Finished neighborhood assessment for model={model_name}, dataset={dataset_name}")

    if not all_metrics:
        return pd.DataFrame()
    return pd.concat(all_metrics, ignore_index=True)
