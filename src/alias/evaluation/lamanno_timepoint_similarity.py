from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Collection, Mapping

import numpy as np
import pandas as pd

from alias.evaluation.embedding import load_dataset_embedding_artifacts
from alias.util.artifacts import create_evaluation_run_directory, write_metadata
from alias.util.plots.umap_plots import UMAPCellPlotter


@dataclass
class LaMannoTimepointSimilarityConfig:
    output_dir: Path | str
    lineages: Mapping[str, Collection[str]]
    lineage_celltype_order: Mapping[str, Collection[str]] | None = None
    annotation_column: str = "celltype"
    time_column: str = "time"
    similarity_metric: str = "cosine"
    bins: int = 180
    heatmap_vmin: float | None = None
    heatmap_vmax: float | None = None
    model_name_map: dict[str, str] | None = None


def _safe_component(value: Any) -> str:
    cleaned = re.sub(r"\s+", "_", str(value).strip())
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "", cleaned)
    return cleaned or "unknown"


def _embedding_columns(df: pd.DataFrame, excluded_columns: set[str]) -> list[Any]:
    return [
        column
        for column in df.columns
        if column not in excluded_columns and np.issubdtype(df[column].dtype, np.number)
    ]


def _similarity(left: np.ndarray, right: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        left_norm = np.linalg.norm(left, axis=1, keepdims=True)
        right_norm = np.linalg.norm(right, axis=1, keepdims=True)
        left_norm[left_norm == 0] = 1.0
        right_norm[right_norm == 0] = 1.0
        return (left / left_norm) @ (right / right_norm).T
    if metric == "dot":
        return left @ right.T
    raise ValueError(f"Unknown similarity metric: {metric}")


def _load_adata(eval_data: Any) -> Any:
    if isinstance(eval_data, dict) and eval_data.get("adata") is not None:
        return eval_data["adata"].copy()
    raise ValueError("LaManno timepoint similarity requires `eval_data['adata']`.")


def _adata_path(eval_data: Any) -> str | None:
    if isinstance(eval_data, dict) and eval_data.get("adata_path") is not None:
        return str(eval_data["adata_path"])
    return None


def _canonical_timepoint_label(value: Any) -> str:
    try:
        return str(float(value))
    except (TypeError, ValueError):
        return str(value)


def _timepoint_embeddings_from_additional(
    loaded_artifacts: dict[str, dict[str, Any]],
    *,
    timepoint_labels: list[str],
    saved_model_name: str,
    dataset_name: str,
) -> tuple[np.ndarray, str | None]:
    additional_artifact = loaded_artifacts.get("df_additional")
    if additional_artifact is None:
        raise ValueError(
            "LaManno timepoint similarity requires saved `df_additional` embeddings "
            f"for timepoint labels {timepoint_labels} in {saved_model_name}/{dataset_name}. "
            "Regenerate embeddings with GenEmbeddingsConfig(additional_data=timepoint_labels)."
        )

    df_additional = additional_artifact["dataframe"].copy()
    label_column = "data" if "data" in df_additional.columns else None
    if label_column is None:
        raise ValueError(
            f"`df_additional` for {saved_model_name}/{dataset_name} is missing the `data` label column."
        )

    embedding_columns = _embedding_columns(df_additional, excluded_columns={label_column})
    if not embedding_columns:
        raise ValueError(
            f"`df_additional` for {saved_model_name}/{dataset_name} does not contain numeric embeddings."
        )

    df_additional["_timepoint_label"] = df_additional[label_column].map(_canonical_timepoint_label)
    missing = sorted(set(timepoint_labels).difference(set(df_additional["_timepoint_label"])))
    if missing:
        raise ValueError(
            f"`df_additional` for {saved_model_name}/{dataset_name} is missing timepoint labels {missing}. "
            "Regenerate embeddings with the full set of LaManno timepoint strings in additional_data."
        )

    indexed = df_additional.drop_duplicates("_timepoint_label").set_index("_timepoint_label")
    return (
        indexed.loc[timepoint_labels, embedding_columns].to_numpy(dtype=np.float32),
        additional_artifact["metadata"].get("path"),
    )


def _align_cells_to_adata(
    df_cells: pd.DataFrame,
    adata,
    *,
    annotation_column: str,
    time_column: str,
    dataset_name: str,
) -> pd.DataFrame:
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    df_cells = df_cells.copy()
    df_cells.index = df_cells.index.astype(str)

    for column in (annotation_column, time_column):
        if column not in obs.columns:
            raise ValueError(f"AnnData observations are missing `{column}` for {dataset_name}.")

    missing_cells = df_cells.index.difference(obs.index)
    if len(missing_cells) > 0:
        examples = ", ".join(map(str, missing_cells[:5]))
        raise ValueError(
            f"Embedding cells are not aligned to adata.obs for {dataset_name}. "
            f"Missing {len(missing_cells)} cells, examples: {examples}"
        )

    df_cells[annotation_column] = obs.loc[df_cells.index, annotation_column].astype(str)
    df_cells[time_column] = pd.to_numeric(obs.loc[df_cells.index, time_column], errors="coerce")
    return df_cells


def _lineage_scores(
    df_lineage: pd.DataFrame,
    *,
    embedding_columns: list[Any],
    timepoint_labels: list[str],
    time_embeddings: np.ndarray,
    annotation_column: str,
    time_column: str,
    lineage_name: str,
    saved_model_name: str,
    dataset_name: str,
    metric: str,
) -> pd.DataFrame:
    cell_embeddings = df_lineage[embedding_columns].to_numpy(dtype=np.float32)
    scores = _similarity(cell_embeddings, time_embeddings, metric)
    descending_order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(descending_order)
    row_indices = np.arange(scores.shape[0])[:, None]
    ranks[row_indices, descending_order] = np.arange(1, scores.shape[1] + 1)
    observed_timepoint_labels = [str(float(value)) for value in df_lineage[time_column]]
    target_lookup = {label: index for index, label in enumerate(timepoint_labels)}
    true_timepoint_ranks = [
        int(ranks[cell_index, target_lookup[label]]) if label in target_lookup else np.nan
        for cell_index, label in enumerate(observed_timepoint_labels)
    ]
    true_timepoint_similarities = [
        float(scores[cell_index, target_lookup[label]]) if label in target_lookup else np.nan
        for cell_index, label in enumerate(observed_timepoint_labels)
    ]
    best_timepoints = [
        timepoint_labels[int(descending_order[cell_index, 0])]
        for cell_index in range(scores.shape[0])
    ]

    score_frames = []
    base = pd.DataFrame(
        {
            "cell_id": df_lineage.index.astype(str),
            annotation_column: df_lineage[annotation_column].astype(str).to_numpy(),
            "observed_time": df_lineage[time_column].to_numpy(dtype=float),
            "observed_timepoint_label": observed_timepoint_labels,
            "best_timepoint": best_timepoints,
            "true_timepoint_rank": true_timepoint_ranks,
            "true_timepoint_similarity": true_timepoint_similarities,
        }
    )
    for index, timepoint_label in enumerate(timepoint_labels):
        df_score = base.copy()
        df_score["lineage"] = lineage_name
        df_score["target_timepoint"] = timepoint_label
        df_score["sim_score"] = scores[:, index]
        df_score["is_observed_timepoint"] = (
            df_score["observed_timepoint_label"] == timepoint_label
        )
        df_score["model_name"] = saved_model_name
        df_score["dataset_name"] = dataset_name
        score_frames.append(df_score)

    if not score_frames:
        return pd.DataFrame()
    return pd.concat(score_frames, ignore_index=True)


def _summary(results: pd.DataFrame, annotation_column: str) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    group_columns = [
        "model_name",
        "dataset_name",
        "lineage",
        "target_timepoint",
        "observed_time",
        annotation_column,
    ]
    return (
        results.groupby(group_columns, dropna=False, sort=True)["sim_score"]
        .agg(n_cells="size", mean_similarity="mean", median_similarity="median")
        .reset_index()
    )


def _config_metadata(config: LaMannoTimepointSimilarityConfig) -> dict[str, Any]:
    metadata = asdict(config)
    metadata["output_dir"] = str(config.output_dir)
    metadata["lineages"] = {
        str(key): sorted(map(str, values))
        for key, values in config.lineages.items()
    }
    if config.lineage_celltype_order is not None:
        metadata["lineage_celltype_order"] = {
            str(key): list(map(str, values))
            for key, values in config.lineage_celltype_order.items()
        }
    return metadata


def _plot_lineage_histograms(
    lineage_dir: Path,
    lineage_results: pd.DataFrame,
    *,
    annotation_column: str,
    bins: int,
) -> dict[str, str]:
    plotter = UMAPCellPlotter()
    hist_dir = lineage_dir / "hist"
    paths: dict[str, str] = {}
    for timepoint_label in sorted(lineage_results["target_timepoint"].unique(), key=lambda value: float(value)):
        df_plot = lineage_results[lineage_results["target_timepoint"] == timepoint_label]
        output_path = hist_dir / f"{_safe_component(timepoint_label)}.pdf"
        plotter.plot_similarity_histogram_grouped(
            df=df_plot,
            annotation_column=annotation_column,
            bins=bins,
            output_path=output_path,
            title=f"{lineage_dir.name}: timepoint {timepoint_label}",
        )
        paths[str(timepoint_label)] = str(output_path)
    return paths


def _plot_lineage_heatmap(
    lineage_dir: Path,
    lineage_results: pd.DataFrame,
    *,
    vmin: float | None,
    vmax: float | None,
) -> str | None:
    if lineage_results.empty:
        return None

    heatmap_df = lineage_results.pivot_table(
        index="observed_timepoint_label",
        columns="target_timepoint",
        values="sim_score",
        aggfunc="mean",
    )
    if heatmap_df.empty:
        return None

    ordered_rows = sorted(heatmap_df.index, key=lambda value: float(value))
    ordered_columns = sorted(heatmap_df.columns, key=lambda value: float(value))
    heatmap_df = heatmap_df.loc[ordered_rows, ordered_columns]

    output_path = lineage_dir / "observed_time_target_timepoint_heatmap.pdf"
    highlight_cells = {
        timepoint_label: [timepoint_label]
        for timepoint_label in ordered_columns
        if timepoint_label in set(ordered_rows)
    }
    UMAPCellPlotter(colormap_name="magma").plot_similarity_heatmap(
        sim_df=heatmap_df,
        output_path=output_path,
        title=f"{lineage_dir.name}: observed time vs target timepoint",
        row_labels=ordered_rows,
        col_labels=ordered_columns,
        x_label="Target timepoint",
        y_label="Observed timepoint",
        vmin=vmin,
        vmax=vmax,
        highlight_cells=highlight_cells,
    )
    return str(output_path)


def _plot_lineage_celltype_heatmap(
    lineage_dir: Path,
    lineage_results: pd.DataFrame,
    *,
    lineage_name: str,
    annotation_column: str,
    lineage_celltype_order: Mapping[str, Collection[str]] | None,
    vmin: float | None,
    vmax: float | None,
) -> str | None:
    if lineage_results.empty:
        return None

    heatmap_df = lineage_results.pivot_table(
        index=annotation_column,
        columns="target_timepoint",
        values="sim_score",
        aggfunc="mean",
    )
    if heatmap_df.empty:
        return None

    present_rows = list(map(str, heatmap_df.index))
    configured_order = (
        list(map(str, lineage_celltype_order.get(lineage_name, [])))
        if lineage_celltype_order is not None
        else []
    )
    ordered_rows = [celltype for celltype in configured_order if celltype in set(present_rows)]
    ordered_rows.extend(sorted(set(present_rows).difference(ordered_rows)))
    ordered_columns = sorted(heatmap_df.columns, key=lambda value: float(value))
    heatmap_df = heatmap_df.loc[ordered_rows, ordered_columns]

    output_path = lineage_dir / "celltype_target_timepoint_heatmap.pdf"
    UMAPCellPlotter(colormap_name="magma").plot_similarity_heatmap(
        sim_df=heatmap_df,
        output_path=output_path,
        title=f"{lineage_dir.name}: cell type vs target timepoint",
        row_labels=ordered_rows,
        col_labels=ordered_columns,
        x_label="Target timepoint",
        y_label="Cell type",
        vmin=vmin,
        vmax=vmax,
    )
    return str(output_path)


def lamanno_timepoint_similarity(
    embeddings_dict: dict,
    eval_data: Any,
    config: LaMannoTimepointSimilarityConfig,
    model: Any | None = None,
) -> pd.DataFrame:
    adata = _load_adata(eval_data)
    all_results: list[pd.DataFrame] = []

    for saved_model_name, model_data in embeddings_dict.items():
        for dataset_name, dataset_meta in model_data.items():
            loaded_dataset = load_dataset_embedding_artifacts(
                dataset_meta,
                annotation_column=config.annotation_column,
            )
            loaded_artifacts = loaded_dataset["artifacts"]
            cell_artifact = loaded_artifacts.get("df_cells")
            if cell_artifact is None:
                continue

            run_dir = create_evaluation_run_directory(
                output_dir=config.output_dir,
                model_name=saved_model_name,
                dataset_name=dataset_name,
                evaluation_name="lamanno_timepoint_similarity",
                timestamp=loaded_dataset["run_timestamp"],
            )

            df_cells = _align_cells_to_adata(
                cell_artifact["dataframe"],
                adata,
                annotation_column=config.annotation_column,
                time_column=config.time_column,
                dataset_name=dataset_name,
            )
            embedding_columns = _embedding_columns(
                df_cells,
                excluded_columns={config.annotation_column, config.time_column},
            )
            if not embedding_columns:
                raise ValueError(f"No numeric cell embedding columns found for {saved_model_name}/{dataset_name}.")

            valid_times = sorted(
                df_cells[config.time_column].dropna().unique(),
                key=lambda value: float(value),
            )
            timepoint_labels = [str(float(value)) for value in valid_times]
            time_embeddings, timepoint_embedding_path = _timepoint_embeddings_from_additional(
                loaded_artifacts,
                timepoint_labels=timepoint_labels,
                saved_model_name=saved_model_name,
                dataset_name=dataset_name,
            )

            lineage_plot_paths: dict[str, dict[str, str]] = {}
            heatmap_paths: dict[str, str] = {}
            celltype_heatmap_paths: dict[str, str] = {}
            dataset_results: list[pd.DataFrame] = []
            for lineage_name, lineage_celltypes in config.lineages.items():
                lineage_dir = run_dir / _safe_component(lineage_name)
                lineage_dir.mkdir(parents=True, exist_ok=True)
                df_lineage = df_cells[
                    df_cells[config.annotation_column].isin({str(value) for value in lineage_celltypes})
                ].copy()
                if df_lineage.empty:
                    continue

                lineage_results = _lineage_scores(
                    df_lineage,
                    embedding_columns=embedding_columns,
                    timepoint_labels=timepoint_labels,
                    time_embeddings=time_embeddings,
                    annotation_column=config.annotation_column,
                    time_column=config.time_column,
                    lineage_name=str(lineage_name),
                    saved_model_name=saved_model_name,
                    dataset_name=dataset_name,
                    metric=config.similarity_metric,
                )
                lineage_results.to_csv(lineage_dir / "scores.csv", index=False)
                lineage_plot_paths[str(lineage_name)] = _plot_lineage_histograms(
                    lineage_dir=lineage_dir,
                    lineage_results=lineage_results,
                    annotation_column=config.annotation_column,
                    bins=config.bins,
                )
                heatmap_path = _plot_lineage_heatmap(
                    lineage_dir,
                    lineage_results,
                    vmin=config.heatmap_vmin,
                    vmax=config.heatmap_vmax,
                )
                if heatmap_path is not None:
                    heatmap_paths[str(lineage_name)] = heatmap_path
                celltype_heatmap_path = _plot_lineage_celltype_heatmap(
                    lineage_dir,
                    lineage_results,
                    lineage_name=str(lineage_name),
                    annotation_column=config.annotation_column,
                    lineage_celltype_order=config.lineage_celltype_order,
                    vmin=config.heatmap_vmin,
                    vmax=config.heatmap_vmax,
                )
                if celltype_heatmap_path is not None:
                    celltype_heatmap_paths[str(lineage_name)] = celltype_heatmap_path
                dataset_results.append(lineage_results)

            if dataset_results:
                combined = pd.concat(dataset_results, ignore_index=True)
            else:
                combined = pd.DataFrame()
            summary = _summary(combined, config.annotation_column)

            results_path = run_dir / "results_df.csv"
            summary_path = run_dir / "timepoint_similarity_summary.csv"
            combined.to_csv(results_path, index=False)
            summary.to_csv(summary_path, index=False)
            write_metadata(
                run_dir,
                {
                    "evaluation_name": "lamanno_timepoint_similarity",
                    "model_name": saved_model_name,
                    "dataset_name": dataset_name,
                    "run_timestamp": run_dir.name,
                    "embedding_run_timestamp": loaded_dataset["run_timestamp"],
                    "embedding_run_dir": cell_artifact["metadata"].get("run_dir"),
                    "embedding_path": cell_artifact["metadata"].get("path"),
                    "timepoint_embedding_source": "df_additional",
                    "timepoint_embedding_path": timepoint_embedding_path,
                    "adata_path": _adata_path(eval_data),
                    "annotation_column": config.annotation_column,
                    "time_column": config.time_column,
                    "similarity_metric": config.similarity_metric,
                    "bins": config.bins,
                    "heatmap_vmin": config.heatmap_vmin,
                    "heatmap_vmax": config.heatmap_vmax,
                    "heatmap_colormap": "magma",
                    "lineages": {
                        str(key): sorted(map(str, values))
                        for key, values in config.lineages.items()
                    },
                    "lineage_celltype_order": {
                        str(key): list(map(str, values))
                        for key, values in (config.lineage_celltype_order or {}).items()
                    },
                    "timepoint_labels": timepoint_labels,
                    "results_path": str(results_path),
                    "summary_path": str(summary_path),
                    "histogram_paths": lineage_plot_paths,
                    "heatmap_paths": heatmap_paths,
                    "celltype_heatmap_paths": celltype_heatmap_paths,
                    "n_rows": int(len(combined)),
                    "config": _config_metadata(config),
                },
            )
            if not combined.empty:
                all_results.append(combined)

    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)
