from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns

from alias.evaluation.celltype_label_plots import EvaluationConfig, compute_umap
from alias.evaluation.embedding import load_dataset_embedding_artifacts, load_embedding_model
from alias.util.artifacts import create_evaluation_run_directory, write_metadata
from alias.util.plots.color_definition import COLOR_GROUPS
from alias.util.plots.umap_plots import UMAPCellPlotter
from alias.util.similarity import evaluate_similarity_meta


TAB20B_RED = plt.get_cmap("tab20b").colors[12]
TAB20C_DARK_RED = TAB20B_RED
TAB20C_LIGHT_RED = COLOR_GROUPS["slate"][-1]
TAB20C_LIGHT_ORANGE = COLOR_GROUPS["orange"][-1]
TAB20C_DARK_ORANGE = COLOR_GROUPS["orange"][0]
DEG_COMPARISON_COLORS = [TAB20C_DARK_ORANGE, TAB20C_LIGHT_ORANGE]


@dataclass
class DiseaseComparisonConfig:
    disease_column: str = "subject.cmv"
    disease_strings: list[str] = field(
        default_factory=lambda: ["increased cytotoxic activity in cytomegalovirus positive patients"]
    )
    output_dir: Path = Path(".")
    adata_path: Path | str | None = None
    model_name: str | None = None
    similarity_metric: str = "cosine"
    bins: int = 60
    positive_values: list[Any] | None = None


def _load_adata(eval_data: Any, config: DiseaseComparisonConfig):
    if isinstance(eval_data, dict) and eval_data.get("adata") is not None:
        return eval_data["adata"].copy()
    if config.adata_path is not None:
        return sc.read_h5ad(config.adata_path)
    raise ValueError("Disease comparison requires `eval_data['adata']` or `config.adata_path`.")


def _resolve_model_name(model_name: str, config: DiseaseComparisonConfig, layers_config: Any) -> str:
    if config.model_name is not None:
        return config.model_name
    if layers_config is not None and getattr(layers_config, "model", None) is not None:
        return layers_config.model
    return model_name


def _is_positive(value: Any, positive_values: list[Any] | None) -> bool:
    if pd.isna(value):
        return False
    if positive_values is not None:
        return value in positive_values
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    normalized = str(value).strip().lower()
    return normalized in {"positive", "true", "1", "yes"}


def _align_cell_metadata(df_cells: pd.DataFrame, adata, annotation_column: str, config: DiseaseComparisonConfig) -> pd.DataFrame:
    adata.obs.index = adata.obs.index.astype(str)
    shared_index = df_cells.index.intersection(adata.obs.index)
    if shared_index.empty:
        raise ValueError("No shared cell ids between saved embeddings and AnnData observations.")

    aligned = df_cells.loc[shared_index].copy()
    aligned[annotation_column] = adata.obs.loc[shared_index, annotation_column].values
    aligned[config.disease_column] = adata.obs.loc[shared_index, config.disease_column].values
    aligned["disease_positive"] = adata.obs.loc[shared_index, config.disease_column].map(
        lambda value: _is_positive(value, config.positive_values)
    ).astype(bool)
    return aligned


def _extract_embedding_frame(df: pd.DataFrame, metadata_columns: set[str]) -> pd.DataFrame:
    candidate_columns = [column for column in df.columns if column not in metadata_columns]
    embedding_only = df[candidate_columns].select_dtypes(include=[np.number]).copy()
    if embedding_only.empty:
        raise ValueError("No numeric embedding columns found.")
    embedding_only["embedding"] = embedding_only.to_numpy(dtype=np.float32).tolist()
    return embedding_only


def _prepare_umap_frames(
    df_cells: pd.DataFrame,
    df_centroids: pd.DataFrame,
    cell_umap: pd.DataFrame | None,
    centroid_umap: pd.DataFrame | None,
    annotation_column: str,
    evaluation_config: EvaluationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if cell_umap is not None and centroid_umap is not None:
        df_cells_umap = df_cells.copy()
        df_cells_umap[["UMAP1", "UMAP2"]] = cell_umap.loc[df_cells_umap.index, ["UMAP1", "UMAP2"]].to_numpy()

        centroid_key = "cell_type" if "cell_type" in centroid_umap.columns else annotation_column
        centroid_positions = centroid_umap.set_index(centroid_key)[["UMAP1", "UMAP2"]]
        df_centroids_umap = pd.DataFrame(
            {
                "cell_type": df_centroids[annotation_column].values,
                "UMAP1": centroid_positions.loc[df_centroids[annotation_column], "UMAP1"].to_numpy(),
                "UMAP2": centroid_positions.loc[df_centroids[annotation_column], "UMAP2"].to_numpy(),
            }
        )
        return df_cells_umap, df_centroids_umap

    metadata_columns = {annotation_column, "disease_positive", "UMAP1", "UMAP2"}
    cell_embeddings = _extract_embedding_frame(df_cells, metadata_columns=metadata_columns).drop(columns=["embedding"]).to_numpy(
        dtype=np.float32
    )
    centroid_embeddings = _extract_embedding_frame(df_centroids, metadata_columns={annotation_column}).drop(
        columns=["embedding"]
    ).to_numpy(dtype=np.float32)

    combined_embeddings = np.vstack([cell_embeddings, centroid_embeddings])
    combined_umap = compute_umap(combined_embeddings, evaluation_config)

    df_cells_umap = df_cells.copy()
    df_cells_umap[["UMAP1", "UMAP2"]] = combined_umap[: len(df_cells_umap)]

    df_centroids_umap = pd.DataFrame(
        {
            "cell_type": df_centroids[annotation_column].values,
            "UMAP1": combined_umap[len(df_cells_umap) :, 0],
            "UMAP2": combined_umap[len(df_cells_umap) :, 1],
        }
    )
    return df_cells_umap, df_centroids_umap


def _load_disease_embeddings(
    loaded_artifacts: dict[str, dict[str, Any]],
    disease_strings: list[str],
    model_name: str | None = None,
) -> list[tuple[int, str, np.ndarray]]:
    additional_artifact = loaded_artifacts.get("df_additional")
    if additional_artifact is None:
        if model_name is None:
            raise ValueError("Disease comparison requires saved `df_additional` embeddings` or a model name.")
        model = load_embedding_model(model_name)
        encoded = np.asarray(model.encode(disease_strings), dtype=np.float32)
        return [
            (disease_idx, disease_str, encoded[disease_idx - 1].reshape(1, -1))
            for disease_idx, disease_str in enumerate(disease_strings, start=1)
        ]

    df_additional = additional_artifact["dataframe"].copy()
    label_column = "data" if "data" in df_additional.columns else None
    if label_column is None:
        raise ValueError("Saved `df_additional` artifact is missing the `data` mapping column.")

    embedding_columns = [column for column in df_additional.columns if column != label_column]
    embedding_frame = df_additional[embedding_columns].select_dtypes(include=[np.number])
    if embedding_frame.empty:
        raise ValueError("Saved `df_additional` artifact does not contain numeric embedding columns.")

    by_label = {
        label: embedding_frame.iloc[idx].to_numpy(dtype=np.float32).reshape(1, -1)
        for idx, label in enumerate(df_additional[label_column].astype(str).tolist())
    }

    missing = [disease_str for disease_str in disease_strings if disease_str not in by_label]
    if missing:
        raise ValueError(f"Missing saved disease embeddings for: {missing}")

    return [
        (disease_idx, disease_str, by_label[disease_str])
        for disease_idx, disease_str in enumerate(disease_strings, start=1)
    ]


def _plot_reference_umaps(
    df_cells_umap: pd.DataFrame,
    df_centroids_umap: pd.DataFrame,
    adata,
    annotation_column: str,
    config: DiseaseComparisonConfig,
    output_dir: Path,
) -> None:
    umap_df_adata = pd.DataFrame(
        {
            "UMAP1": adata.obsm["X_umap"][:, 0],
            "UMAP2": adata.obsm["X_umap"][:, 1],
            "disease_positive": adata.obs[config.disease_column].map(
                lambda value: _is_positive(value, config.positive_values)
            ).astype(bool),
            annotation_column: adata.obs[annotation_column].values,
        }
    )

    for color_column in [annotation_column, "disease_positive"]:
        plotter = UMAPCellPlotter(palette_name="tabc2")
        plotter.annotate_centroids = False
        plotter.plot_cells(
            df_cells_umap,
            annotation_column=color_column,
            output_path=output_dir / f"cells_colored_by_{color_column}_embeddings.pdf",
            annotate_centroids_df=df_centroids_umap,
            title=f"Cells colored by {color_column}",
        )
        plotter.plot_cells(
            umap_df_adata,
            annotation_column=color_column,
            output_path=output_dir / f"cells_colored_by_{color_column}_adata.pdf",
            title=f"Cells colored by {color_column}",
        )


def _run_deg_analysis(adata, celltype_dir: Path, disease_column: str) -> None:
    group_counts = adata.obs["associated"].value_counts()
    if not all(group_counts.get(val, 0) > 30 for val in [True, False]):
        print("Skipping DEG analysis: not enough cells in one or both groups.")
        return

    adata_sub = adata.copy()
    colors_for_groups = ["#1f77b4", "#aec7e8"]
    sc.settings.figdir = str(celltype_dir)

    adata_sub.obs["associated"] = adata_sub.obs["associated"].map({True: "True", False: "False"}).astype("category")
    adata_sub.uns["associated_colors"] = colors_for_groups
    sc.tl.rank_genes_groups(adata_sub, groupby="associated", reference="False", method="wilcoxon")
    top_genes_df = sc.get.rank_genes_groups_df(adata_sub, group="True")
    sig_genes_df = top_genes_df[top_genes_df["pvals_adj"] < 0.05]
    top5_up = sig_genes_df[sig_genes_df["logfoldchanges"] > 0].head(5)["names"].tolist()
    top5_down = sig_genes_df[sig_genes_df["logfoldchanges"] < 0].head(5)["names"].tolist()

    if top5_up:
        sc.pl.rank_genes_groups_heatmap(
            adata_sub,
            groups=["True", "False"],
            var_names=top5_up,
            dendrogram=False,
            show=False,
            save="_associated.pdf",
        )

    sc.tl.rank_genes_groups(adata_sub, groupby=disease_column, reference="Negative", method="wilcoxon")
    top_genes_df_meta = sc.get.rank_genes_groups_df(adata_sub, group="Positive")
    sig_genes_meta_df = top_genes_df_meta[top_genes_df_meta["pvals_adj"] < 0.05]
    top5_up_meta = sig_genes_meta_df[sig_genes_meta_df["logfoldchanges"] > 0].head(5)["names"].tolist()
    top5_down_meta = sig_genes_meta_df[sig_genes_meta_df["logfoldchanges"] < 0].head(5)["names"].tolist()

    if top5_up_meta:
        adata_sub.uns[f"{disease_column}_colors"] = colors_for_groups
        sc.pl.rank_genes_groups_heatmap(
            adata_sub,
            groups=["Positive", "Negative"],
            var_names=top5_up_meta,
            dendrogram=False,
            show=False,
            save="_metadata.pdf",
        )

    overlap_genes = set(top5_up + top5_down).intersection(set(top5_up_meta + top5_down_meta))
    if not overlap_genes:
        return

    df_method1 = top_genes_df[top_genes_df["names"].isin(overlap_genes)]
    df_method2 = top_genes_df_meta[top_genes_df_meta["names"].isin(overlap_genes)]
    df_compare = df_method1[["names", "logfoldchanges", "pvals_adj"]].merge(
        df_method2[["names", "logfoldchanges", "pvals_adj"]],
        on="names",
        suffixes=("_assoc", "_meta"),
    )

    palette = DEG_COMPARISON_COLORS

    df_long_logfc = pd.melt(
        df_compare[["names", "logfoldchanges_assoc", "logfoldchanges_meta"]],
        id_vars="names",
        var_name="method",
        value_name="logFC",
    )
    df_long_logfc["method"] = df_long_logfc["method"].map(
        {"logfoldchanges_assoc": "Model", "logfoldchanges_meta": "Metadata"}
    )

    plt.figure(figsize=(3, 2))
    ax = sns.barplot(data=df_long_logfc, x="names", y="logFC", hue="method", palette=palette)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.xticks(rotation=45, ha="right")
    plt.title("Log fold change comparison per gene")
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, frameon=False)
    plt.tight_layout()
    plt.savefig(celltype_dir / "logFC_barplot.pdf", bbox_inches="tight")
    plt.savefig(celltype_dir / "logFC_barplot.svg", bbox_inches="tight")
    plt.close()

    df_long_pval = pd.melt(
        df_compare[["names", "pvals_adj_assoc", "pvals_adj_meta"]],
        id_vars="names",
        var_name="method",
        value_name="pval",
    )
    df_long_pval["method"] = df_long_pval["method"].map(
        {"pvals_adj_assoc": "Model", "pvals_adj_meta": "Metadata"}
    )
    df_long_pval["minus_log10_pval"] = -np.log10(df_long_pval["pval"] + 1e-300)

    plt.figure(figsize=(3, 2))
    ax = sns.barplot(
        data=df_long_pval,
        x="names",
        y="minus_log10_pval",
        hue="method",
        palette=palette,
    )
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("-log10(p-value)")
    plt.title("P-value comparison per gene")
    ax.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, frameon=False)
    plt.tight_layout()
    plt.savefig(celltype_dir / "pvalue_barplot.pdf", bbox_inches="tight")
    plt.savefig(celltype_dir / "pvalue_barplot.svg", bbox_inches="tight")
    plt.close()


def disease_comparison(
    embeddings_dict: dict,
    eval_data: Any | None = None,
    subfolder_fig_dir: str | Path | None = None,
    annotation_column: str | None = None,
    evaluation_config: EvaluationConfig | None = None,
    layers_config: Any | None = None,
    config: DiseaseComparisonConfig | None = None,
) -> pd.DataFrame:
    """
    Compare a disease prompt against cells within each cell type.

    Supports the legacy call shape while using the current embedding artifact metadata.
    """
    if annotation_column is None:
        raise ValueError("`annotation_column` is required.")
    if evaluation_config is None:
        raise ValueError("`evaluation_config` is required.")

    config = config or DiseaseComparisonConfig()
    if subfolder_fig_dir is not None:
        config.output_dir = Path(subfolder_fig_dir)

    all_results: list[pd.DataFrame] = []

    for saved_model_name, model_data in embeddings_dict.items():
        for dataset_name, dataset_meta in model_data.items():
            resolved_model_name = _resolve_model_name(saved_model_name, config, layers_config)
            loaded_dataset = load_dataset_embedding_artifacts(dataset_meta, annotation_column=annotation_column)
            loaded_artifacts = loaded_dataset["artifacts"]
            if "df_cells" not in loaded_artifacts or "df_celltypes" not in loaded_artifacts:
                continue

            adata = _load_adata(eval_data, config)
            cell_artifact = loaded_artifacts["df_cells"]
            centroid_artifact = loaded_artifacts["df_celltypes"]

            df_cells = _align_cell_metadata(cell_artifact["dataframe"], adata, annotation_column, config)
            df_centroids = centroid_artifact["dataframe"].copy()
            if annotation_column not in df_centroids.columns and "cell_type" in df_centroids.columns:
                df_centroids[annotation_column] = df_centroids["cell_type"]
            if annotation_column not in df_centroids.columns:
                raise ValueError(f"Centroid embeddings are missing `{annotation_column}` annotations.")

            df_cells_umap, df_centroids_umap = _prepare_umap_frames(
                df_cells=df_cells,
                df_centroids=df_centroids,
                cell_umap=cell_artifact["umap"],
                centroid_umap=centroid_artifact["umap"],
                annotation_column=annotation_column,
                evaluation_config=evaluation_config,
            )

            run_dir = create_evaluation_run_directory(
                output_dir=config.output_dir,
                model_name=saved_model_name,
                dataset_name=dataset_name,
                evaluation_name="disease_comparison",
                timestamp=loaded_dataset["run_timestamp"],
            )

            _plot_reference_umaps(
                df_cells_umap=df_cells_umap,
                df_centroids_umap=df_centroids_umap,
                adata=adata,
                annotation_column=annotation_column,
                config=config,
                output_dir=run_dir,
            )

            df_cells_features = _extract_embedding_frame(
                df_cells_umap,
                metadata_columns={annotation_column, config.disease_column, "disease_positive", "UMAP1", "UMAP2"},
            )
            df_cells_work = df_cells_umap[["UMAP1", "UMAP2", annotation_column, config.disease_column, "disease_positive"]].copy()
            df_cells_work["embedding"] = df_cells_features["embedding"]

            centroid_features = _extract_embedding_frame(df_centroids, metadata_columns={annotation_column})
            df_centroids_work = df_centroids_umap.copy()
            df_centroids_work["embedding"] = centroid_features["embedding"]
            disease_embeddings = _load_disease_embeddings(
                loaded_artifacts,
                config.disease_strings,
                model_name=resolved_model_name,
            )

            dataset_results: list[dict[str, Any]] = []
            for disease_idx, disease_str, disease_emb in disease_embeddings:
                disease_dir = run_dir / f"disease_only_{disease_idx:02d}"
                disease_dir.mkdir(parents=True, exist_ok=True)

                for celltype in sorted(df_cells_work[annotation_column].dropna().unique()):
                    df_cells_sub = df_cells_work[df_cells_work[annotation_column] == celltype].copy()
                    df_centroids_sub = df_centroids_work[df_centroids_work["cell_type"] == celltype].copy()
                    if df_cells_sub.empty or df_centroids_sub.empty:
                        continue
                    if df_cells_sub["disease_positive"].nunique() < 2:
                        print(f"Skipping {celltype}: disease labels do not contain both classes.")
                        continue

                    celltype_dir = disease_dir / f"{str(celltype).replace('/', '_')}"
                    celltype_dir.mkdir(parents=True, exist_ok=True)

                    result = evaluate_similarity_meta(
                        df_cells_sub.rename(columns={"disease_positive": "disease_positive"}),
                        df_centroids_sub,
                        out_dir=celltype_dir,
                        disease_emb=disease_emb,
                        label_key="disease_positive",
                        bins=config.bins,
                        similarity_metric=config.similarity_metric,
                        annotation_column=annotation_column,
                        annotation_column_value=celltype,
                    )
                    result["disease_index"] = disease_idx
                    result["disease_string"] = disease_str
                    dataset_results.append(result)

                    df_sim = result["df_sim"]
                    df_sim.to_csv(celltype_dir / "df_sim.csv", index=True)

                    if "associated" in df_sim.columns:
                        shared_indices = adata.obs.index.intersection(df_sim.index)
                        adata_sub = adata[adata.obs.index.isin(shared_indices)].copy()
                        adata_sub.obs["associated"] = df_sim.loc[adata_sub.obs_names, "associated"]
                        _run_deg_analysis(adata_sub, celltype_dir, config.disease_column)

                if not dataset_results:
                    continue

                df_results = pd.DataFrame(
                    [{key: value for key, value in row.items() if key != "df_sim"} for row in dataset_results]
                )
                cell_counts = df_cells_work.groupby(annotation_column).size().reset_index(name="cell_count")
                df_results = df_results.merge(cell_counts, left_on="cell_type", right_on=annotation_column, how="left")
                df_results["mean_diff"] = df_results["mean_sim_disease"] - df_results["mean_sim_other"]
                df_results["-log10p"] = -np.log10(df_results["mw_p"])
                df_results.to_csv(disease_dir / "results_df.csv", index=False)

                plotter = UMAPCellPlotter(point_size=25)
                plotter.plot_distribution_difference(
                    df=df_results,
                    x_column="mean_diff",
                    y_column="-log10p",
                    label_column="cell_type",
                    pval_column="mw_p",
                    count_column="cell_count",
                    size_scale=7.0,
                    nonsignificant_color=TAB20C_DARK_RED,
                    significant_color=TAB20C_LIGHT_RED,
                    output_path=disease_dir / "distribution_difference_summary.pdf",
                )
                plotter.plot_distribution_difference(
                    df=df_results,
                    x_column="mean_diff",
                    y_column="-log10p",
                    label_column="cell_type",
                    pval_column="mw_p",
                    count_column="cell_count",
                    size_scale=7.0,
                    nonsignificant_color=TAB20C_DARK_RED,
                    significant_color=TAB20C_LIGHT_RED,
                    output_path=disease_dir / "distribution_difference_summary.svg",
                )
                all_results.append(df_results)

            write_metadata(
                run_dir,
                {
                    "evaluation_name": "disease_comparison",
                    "dataset_name": dataset_name,
                    "model_name": saved_model_name,
                    "run_timestamp": run_dir.name,
                    "embedding_run_dir": cell_artifact["metadata"].get("run_dir"),
                    "cell_umap_path": cell_artifact["metadata"].get("umap", {}).get("path"),
                    "celltype_umap_path": centroid_artifact["metadata"].get("umap", {}).get("path"),
                    "disease_column": config.disease_column,
                    "disease_strings": config.disease_strings,
                },
            )

    if not all_results:
        return pd.DataFrame()
    return pd.concat(all_results, ignore_index=True)
