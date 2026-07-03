from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as stats
from scipy.stats import norm, rankdata
from sklearn.decomposition import PCA
import umap

from alias.evaluation.embedding import load_dataset_embedding_artifacts
from alias.util.artifacts import create_evaluation_run_directory, write_metadata
from alias.util.plots.umap_plots import UMAPCellPlotter


LAMANNO_LINEAGES: dict[str, set[str]] = {
    "lineage_1": {
        "Erythroid progenitor",
        "Erythrocyte",
        "Angioblast",
        "Endothelial",
    },
    "lineage_2": {
        "Epiblast",
        "Early ectoderm",
        "Neural crest",
        "Mesenchyme",
        "Early fibroblasts",
        "Intermediate meninges 1",
    },
    "lineage_3": {
        "Hindbrain",
        "Cerebellum glutamatergic",
        "Hindbrain roof plate",
    },
    "lineage_4": {
        "Dorsal forebrain",
        "Forebrain",
        "Ventral forebrain",
        "Neuronal intermediate progenitor",
        "Forebrain glutamatergic",
        "Forebrain GABAergic",
        "Cortical or hippocampal glutamatergic",
    },
}

DEFAULT_MARKERS: list[str] = [
    "Pax6",
    "Emx1",
    "Dlk1",
    "Mest",
    "Dlx1",
    "Gsx2",
    "Eomes",
    "Hes5",
    "Satb2",
    "Bcl11b",
    "Slc1a3",
    "Fabp7",
    "Otx2",
    "Foxg1",
    "Fezf1",
    "Fezf2",
    "Lhx2",
    "Crabp2",
    "Mdk",
    "Ldha",
]


@dataclass
class PseudotimeConfig:
    lineage: str
    cell_origin: str
    output_dir: Path = Path(".")
    markers: list[str] = field(default_factory=lambda: list(DEFAULT_MARKERS))


def _assign_group(annotation_value: Any) -> str | None:
    for group_name, group_set in LAMANNO_LINEAGES.items():
        if annotation_value in group_set:
            return group_name
    return None


def _load_adata(eval_data: Any) -> Any:
    if isinstance(eval_data, dict) and eval_data.get("adata") is not None:
        return eval_data["adata"].copy()
    raise ValueError("Pseudotime requires `eval_data['adata']`.")


def _cal_umap(df_full: pd.DataFrame, evaluation_config: Any) -> pd.DataFrame:
    embedding_array = np.vstack(df_full["embedding"].values)
    pca_model = PCA(n_components=50)
    embedding_pca = pca_model.fit_transform(embedding_array)

    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=evaluation_config.n_neighbors,
        random_state=41,
        min_dist=0.3,
    )
    umap_result = umap_model.fit_transform(embedding_pca)

    df_with_umap = df_full.copy()
    df_with_umap["UMAP1"] = umap_result[:, 0]
    df_with_umap["UMAP2"] = umap_result[:, 1]
    return df_with_umap


def _extract_embedding_columns(df: pd.DataFrame, excluded_columns: set[str]) -> list[Any]:
    return [
        column
        for column in df.columns
        if column not in excluded_columns and np.issubdtype(df[column].dtype, np.number)
    ]


def _kendalltau_log_p(tau: float, n: int) -> float:
    z = tau * np.sqrt(9 * n * (n - 1) / (2 * (2 * n + 5)))
    return norm.logsf(abs(z)) * 2


def _resolve_annotation_column(adata, annotation_column: str) -> str:
    if annotation_column in adata.obs.columns:
        return annotation_column
    if "celltype" in adata.obs.columns:
        return "celltype"
    raise ValueError(f"AnnData observations are missing `{annotation_column}` and `celltype`.")


def _plot_pseudotime_overview(
    lineage_dir: Path,
    expr_ranks: np.ndarray,
    llm_ranks: np.ndarray,
    time_values: np.ndarray,
    cell_types: pd.Series,
    lineage_name: str,
) -> None:
    plotter = UMAPCellPlotter()
    plotter.plot_pseudotime_scatter_time(
        expr_ranks,
        llm_ranks,
        time_values,
        output_path=lineage_dir / f"pseudotime_comparison_time_{lineage_name}.pdf",
    )
    plotter.plot_pseudotime_scatter_celltypes(
        expr_ranks,
        llm_ranks,
        cell_types,
        output_path=lineage_dir / f"pseudotime_comparison_celltypes_{lineage_name}.pdf",
    )


def _plot_lineage_umaps(
    lineage_dir: Path,
    lineage_name: str,
    annotation_column: str,
    df_cells_umap: pd.DataFrame,
    df_centroids_umap: pd.DataFrame,
    df_adata: pd.DataFrame,
    markers_present: list[str],
) -> None:
    plot_columns = ["dpt_pseudotime_llm", "dpt_pseudotime_expr", "time"]
    plotter = UMAPCellPlotter()

    df_adata = df_adata[~df_adata[plot_columns].isna().any(axis=1)]
    df_adata = df_adata[~np.isinf(df_adata[plot_columns]).any(axis=1)]

    for column_name in plot_columns:
        plotter.annotate_centroids = False
        color_vmin = df_adata[column_name].min()
        color_vmax = df_adata[column_name].max()
        plotter.plot_cells(
            df_cells_umap,
            time_color_column=column_name,
            vmin=color_vmin,
            vmax=color_vmax,
            output_path=lineage_dir / f"cells_colored_by_{column_name}_{lineage_name}_llm.pdf",
            title=f"Cells Colored by {column_name} - {lineage_name} - Ours",
        )
        plotter.plot_cells(
            df_adata,
            time_color_column=column_name,
            vmin=color_vmin,
            vmax=color_vmax,
            output_path=lineage_dir / f"cells_colored_by_{column_name}_{lineage_name}_adata.pdf",
            title=f"Cells Colored by {column_name} - {lineage_name} - Adata",
        )

        plotter.annotate_centroids = True
        plotter.plot_cells(
            df_cells_umap,
            time_color_column=column_name,
            annotate_centroids_df=df_centroids_umap,
            vmin=color_vmin,
            vmax=color_vmax,
            output_path=lineage_dir / f"cells_colored_by_{column_name}_{lineage_name}_llm_labels.pdf",
            title=f"Cells Colored by {column_name} - {lineage_name} - Ours with Labels",
        )

        if column_name == "time":
            plotter.plot_cells(
                df_cells_umap,
                annotation_column=annotation_column,
                annotate_centroids_df=df_centroids_umap,
                output_path=lineage_dir / f"cells_colored_by_{annotation_column}_{lineage_name}_llm_labels.pdf",
                title=f"Cells Colored by {annotation_column} - {lineage_name} - Ours with Labels",
            )
            plotter.annotate_centroids = False
            plotter.plot_cells(
                df_adata,
                annotation_column=annotation_column,
                output_path=lineage_dir / f"cells_colored_by_{annotation_column}_{lineage_name}_adata_labels.pdf",
                title=f"Cells Colored by {annotation_column} - {lineage_name} - Adata with Labels",
            )

    for gene_name in markers_present:
        gene_vmin = df_adata[f"{gene_name}_expr"].min()
        gene_vmax = 1 if gene_name == "Fezf1" else df_adata[f"{gene_name}_expr"].max()
        plotter.plot_cells(
            df_cells_umap,
            continuous_color_column=f"{gene_name}_expr",
            vmin=gene_vmin,
            vmax=gene_vmax,
            output_path=lineage_dir / f"cells_colored_by_{gene_name}_{lineage_name}_llm.pdf",
            title=f"Cells Colored by {gene_name} - {lineage_name} - Ours with Labels",
        )
        plotter.plot_cells(
            df_adata,
            continuous_color_column=f"{gene_name}_expr",
            vmin=gene_vmin,
            vmax=gene_vmax,
            output_path=lineage_dir / f"cells_colored_by_{gene_name}_{lineage_name}_adata.pdf",
            title=f"Cells Colored by {gene_name} - {lineage_name} - Adata",
        )


def _run_one_dataset(
    dataset_name: str,
    saved_model_name: str,
    loaded_dataset: dict[str, Any],
    adata,
    annotation_column: str,
    evaluation_config: Any,
    config: PseudotimeConfig,
    run_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    loaded_artifacts = loaded_dataset["artifacts"]
    if "df_cells" not in loaded_artifacts or "df_celltypes" not in loaded_artifacts:
        return pd.DataFrame(), {}

    cell_artifact = loaded_artifacts["df_cells"]
    centroid_artifact = loaded_artifacts["df_celltypes"]

    df_cells = cell_artifact["dataframe"].copy()
    df_centroids = centroid_artifact["dataframe"].copy()

    if annotation_column not in df_centroids.columns and "cell_type" in df_centroids.columns:
        df_centroids[annotation_column] = df_centroids["cell_type"]
    if annotation_column not in df_centroids.columns:
        raise ValueError(f"Centroid embeddings are missing `{annotation_column}` annotations.")

    adata.obs.index = adata.obs.index.astype(str)
    df_cells.index = df_cells.index.astype(str)
    df_centroids.index = df_centroids.index.astype(str)

    resolved_annotation_column = _resolve_annotation_column(adata, annotation_column)
    embedding_columns = _extract_embedding_columns(df_cells, excluded_columns={resolved_annotation_column})

    df_cells["time"] = adata.obs.loc[df_cells.index, "time"].astype(float)
    df_cells["group"] = df_cells[resolved_annotation_column].apply(_assign_group)

    lineage_name = config.lineage
    df_cells_group = df_cells[df_cells["group"] == lineage_name].copy()
    group_celltypes = df_cells_group[resolved_annotation_column].unique().tolist()
    df_centroids_group = df_centroids[df_centroids[annotation_column].isin(group_celltypes)].copy()

    shared_cells = df_cells_group.index.intersection(adata.obs.index)
    df_cells_group = df_cells_group.loc[shared_cells].copy()
    adata_group = adata[shared_cells].copy()

    if "X_umap" not in adata_group.obsm:
        sc.tl.pca(adata_group, svd_solver="arpack")
        sc.pp.neighbors(adata_group, n_neighbors=15, n_pcs=50, use_rep=None)
        sc.tl.umap(adata_group, min_dist=0.5, spread=1.0)

    df_cells_group_ord = df_cells_group.loc[adata_group.obs_names].copy()
    adata_expr = adata_group.copy()

    x_llm = df_cells_group_ord[embedding_columns].to_numpy(dtype=np.float32)
    adata_group.obsm["X_llm"] = x_llm

    sc.pp.neighbors(adata_group, use_rep="X_llm", n_neighbors=15)
    annotation_series = adata_group.obs[resolved_annotation_column]
    root_cells = adata_group.obs_names[annotation_series == config.cell_origin]
    root_cell = adata_group.obs.loc[root_cells].sort_values("time").index[0]
    adata_group.uns["iroot"] = np.where(adata_group.obs_names == root_cell)[0][0]
    sc.tl.dpt(adata_group, copy=False)
    adata_group.obs.rename(columns={"dpt_pseudotime": "dpt_pseudotime_llm"}, inplace=True)

    sc.tl.pca(adata_expr, n_comps=50, svd_solver="arpack")
    sc.pp.neighbors(adata_expr, n_neighbors=15, use_rep="X_pca")
    sc.tl.diffmap(adata_expr)
    adata_expr.uns["iroot"] = np.where(adata_expr.obs_names == root_cell)[0][0]
    sc.tl.dpt(adata_expr, copy=False)
    adata_group.obs["dpt_pseudotime_expr"] = adata_expr.obs["dpt_pseudotime"]

    df_umap = pd.DataFrame(
        adata_group.obsm["X_umap"],
        index=adata_group.obs.index,
        columns=["UMAP1", "UMAP2"],
    )
    df_meta = adata_group.obs[["dpt_pseudotime_llm", "dpt_pseudotime_expr", "time", resolved_annotation_column]]
    df_adata = pd.concat([df_umap, df_meta], axis=1)

    lineage_dir = run_dir / lineage_name
    lineage_dir.mkdir(parents=True, exist_ok=True)

    pt_llm = adata_group.obs["dpt_pseudotime_llm"]
    pt_expr = adata_group.obs["dpt_pseudotime_expr"]
    tau, pval = stats.kendalltau(pt_llm, pt_expr)
    log10_p = None
    if pval == 0.0:
        logp = _kendalltau_log_p(tau, len(pt_llm))
        log10_p = logp / np.log(10)

    expr_ranks = rankdata(adata_group.obs["dpt_pseudotime_expr"])
    llm_ranks = rankdata(adata_group.obs["dpt_pseudotime_llm"])
    time_values = adata_group.obs["time"].values.astype(float)
    cell_types = adata_group.obs[resolved_annotation_column]

    _plot_pseudotime_overview(
        lineage_dir=lineage_dir,
        expr_ranks=expr_ranks,
        llm_ranks=llm_ranks,
        time_values=time_values,
        cell_types=cell_types,
        lineage_name=lineage_name,
    )

    df_cells_group.loc[adata_group.obs_names, "dpt_pseudotime_llm"] = adata_group.obs["dpt_pseudotime_llm"].values
    df_cells_group.loc[adata_group.obs_names, "dpt_pseudotime_expr"] = adata_group.obs["dpt_pseudotime_expr"].values

    df_cells_combined = df_cells_group.copy()
    df_cells_combined["embedding"] = df_cells_combined[embedding_columns].to_numpy(dtype=np.float32).tolist()
    centroid_embedding_columns = _extract_embedding_columns(df_centroids_group, excluded_columns={annotation_column, "cell_type"})
    df_centroids_combined = df_centroids_group.copy()
    df_centroids_combined["embedding"] = df_centroids_group[centroid_embedding_columns].to_numpy(dtype=np.float32).tolist()

    df_combined = pd.concat([df_cells_combined, df_centroids_combined], axis=0)
    df_umap_llm = _cal_umap(df_combined, evaluation_config)
    df_cells_umap = df_umap_llm.iloc[: len(df_cells_group)].copy()
    df_centroids_umap = df_umap_llm.iloc[len(df_cells_group) :].copy()
    df_centroids_umap = df_centroids_umap.rename(columns={annotation_column: "cell_type"})
    adata_group.obsm["X_umap_llm"] = df_cells_umap[["UMAP1", "UMAP2"]].to_numpy()

    adata_path = lineage_dir / "adata_forebrain_llm_20260120.h5ad"
    adata_group.write(adata_path)

    gene_names = adata_group.var_names
    markers_present = [gene for gene in config.markers if gene in gene_names]
    x_matrix = adata_group.X
    if not isinstance(x_matrix, np.ndarray):
        x_matrix = x_matrix.toarray()

    expr_df = pd.DataFrame(
        x_matrix[:, [gene_names.get_loc(gene_name) for gene_name in markers_present]] if markers_present else np.empty((len(adata_group.obs_names), 0)),
        index=adata_group.obs_names,
        columns=[f"{gene_name}_expr" for gene_name in markers_present],
    )

    df_adata = df_adata.join(expr_df, how="left")
    expr_df_aligned = expr_df.reindex(df_cells_umap.index)
    df_cells_umap = df_cells_umap.join(expr_df_aligned)

    _plot_lineage_umaps(
        lineage_dir=lineage_dir,
        lineage_name=lineage_name,
        annotation_column=resolved_annotation_column,
        df_cells_umap=df_cells_umap,
        df_centroids_umap=df_centroids_umap,
        df_adata=df_adata,
        markers_present=markers_present,
    )

    pseudotime_values = adata_group.obs[
        [resolved_annotation_column, "time", "dpt_pseudotime_llm", "dpt_pseudotime_expr"]
    ].copy()
    pseudotime_values.to_csv(lineage_dir / "pseudotime_values.csv")

    metadata = {
        "dataset_name": dataset_name,
        "model_name": saved_model_name,
        "lineage": lineage_name,
        "cell_origin": config.cell_origin,
        "root_cell": root_cell,
        "kendall_tau": float(tau) if pd.notna(tau) else None,
        "kendall_pvalue": float(pval) if pd.notna(pval) else None,
        "kendall_log10_pvalue": float(log10_p) if log10_p is not None else None,
        "lineage_dir": str(lineage_dir),
        "adata_path": str(adata_path),
        "pseudotime_values_path": str(lineage_dir / "pseudotime_values.csv"),
        "n_cells": int(len(pseudotime_values)),
    }
    return pseudotime_values.reset_index().rename(columns={"index": "cell_id"}), metadata


def pseudotime(
    embeddings_dict: dict,
    eval_data: Any | None = None,
    subfolder_fig_dir: str | Path | None = None,
    annotation_column: str | None = None,
    evaluation_config: Any | None = None,
    config: PseudotimeConfig | None = None,
) -> pd.DataFrame:
    if annotation_column is None:
        raise ValueError("`annotation_column` is required.")
    if evaluation_config is None:
        raise ValueError("`evaluation_config` is required.")

    config = config or PseudotimeConfig(lineage="lineage_4", cell_origin="Forebrain")
    if subfolder_fig_dir is not None:
        config.output_dir = Path(subfolder_fig_dir)

    adata = _load_adata(eval_data)
    all_results: list[pd.DataFrame] = []
    metadata_by_dataset: dict[str, Any] = {}

    for saved_model_name, model_data in embeddings_dict.items():
        for dataset_name, dataset_meta in model_data.items():
            loaded_dataset = load_dataset_embedding_artifacts(
                dataset_meta,
                annotation_column=annotation_column,
            )
            run_dir = create_evaluation_run_directory(
                output_dir=config.output_dir,
                model_name=saved_model_name,
                dataset_name=dataset_name,
                evaluation_name="pseudotime",
                timestamp=loaded_dataset["run_timestamp"],
            )
            result_df, lineage_metadata = _run_one_dataset(
                dataset_name=dataset_name,
                saved_model_name=saved_model_name,
                loaded_dataset=loaded_dataset,
                adata=adata.copy(),
                annotation_column=annotation_column,
                evaluation_config=evaluation_config,
                config=config,
                run_dir=run_dir,
            )
            if not result_df.empty:
                all_results.append(result_df)
                metadata_by_dataset[f"{saved_model_name}:{dataset_name}"] = lineage_metadata

            loaded_artifacts = loaded_dataset["artifacts"]
            cell_artifact = loaded_artifacts.get("df_cells", {})
            centroid_artifact = loaded_artifacts.get("df_celltypes", {})
            write_metadata(
                run_dir,
                {
                    "evaluation_name": "pseudotime",
                    "dataset_name": dataset_name,
                    "model_name": saved_model_name,
                    "run_timestamp": run_dir.name,
                    "embedding_run_dir": cell_artifact.get("metadata", {}).get("run_dir"),
                    "cell_umap_path": cell_artifact.get("metadata", {}).get("umap", {}).get("path"),
                    "celltype_umap_path": centroid_artifact.get("metadata", {}).get("umap", {}).get("path"),
                    "lineage": config.lineage,
                    "cell_origin": config.cell_origin,
                    "artifacts": metadata_by_dataset.get(f"{saved_model_name}:{dataset_name}", {}),
                },
            )

    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)
