import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from alias.util.similarity import evaluate_similarity

from alias.util.artifacts import create_evaluation_run_directory, write_metadata
from alias.evaluation.embedding import load_dataset_embedding_artifacts
from alias.util.plots.umap_plots import UMAPCellPlotter


@dataclass
class FunctionalitySimilarityConfig:
    similarity_metric: str = "cosine"
    bins: int = 60
    output_dir: Path = Path(".")
    plot: bool = True
    true_label_map: dict[str, str | list[str]] | None = None


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    left_norm[left_norm == 0] = 1.0
    right_norm[right_norm == 0] = 1.0
    return (left / left_norm) @ (right / right_norm).T


def _similarity_matrix(
    left: np.ndarray,
    right: np.ndarray,
    metric: str,
) -> np.ndarray:
    if metric == "cosine":
        return _cosine_similarity(left, right)
    if metric == "dot":
        return left @ right.T
    raise ValueError(f"Unknown similarity metric: {metric}")


def _plot_heatmap_outputs(
    plotter: UMAPCellPlotter,
    heatmap_df: pd.DataFrame,
    run_dir: Path,
    file_stem: str,
    title: str,
    true_label_map: dict[str, str | list[str]] | None,
) -> dict[str, str]:
    paths = {
        "pdf": run_dir / f"{file_stem}.pdf",
        "png": run_dir / f"{file_stem}.png",
    }
    for output_path in paths.values():
        plotter.plot_similarity_heatmap(
            sim_df=heatmap_df,
            output_path=output_path,
            title=title,
            highlight_cells=true_label_map,
        )
    return {suffix: str(path) for suffix, path in paths.items()}


def functionality_similarity(
    embeddings_dict: dict,
    annotation_column: str,
    config: FunctionalitySimilarityConfig
) -> pd.DataFrame:
    """
    Use 'df_additional' embeddings in embeddings_dict as functionality descriptions
    and compute mean AUC per (functionality, cell type).
    """

    all_results = []

    for model_name, model_data in embeddings_dict.items():
        print(f"Evaluating model: {model_name}")

        for dataset_name, dataset_meta in model_data.items():
            print(f"Processing dataset: {dataset_name}")

            loaded_dataset = load_dataset_embedding_artifacts(
                dataset_meta,
                annotation_column=annotation_column,
            )
            loaded_artifacts = loaded_dataset["artifacts"]

            cell_artifact = loaded_artifacts["df_cells"]
            cell_meta = cell_artifact["metadata"]
            cell_df = cell_artifact["dataframe"]

            if annotation_column not in cell_df.columns:
                raise ValueError(f"{annotation_column} not found in cell dataframe for {dataset_name}")

            cell_embeddings = cell_df.drop(columns=[annotation_column]).values
            cell_annotations = cell_df[annotation_column]
            cell_types = sorted(cell_annotations.unique())
            
            ground_truth = pd.DataFrame({ct: cell_annotations == ct for ct in cell_types})

            cell_umap = cell_artifact["umap"]

            additional_artifact = loaded_artifacts.get("df_additional")
            if additional_artifact is None:
                print(f"No df_additional found for {dataset_name}, skipping")
                continue

            additional_meta = additional_artifact["metadata"]
            df_additional_emb = additional_artifact["dataframe"]
            functionality_column = "data" if "data" in df_additional_emb.columns else annotation_column
            functionality_names = df_additional_emb[functionality_column].tolist()
            functionality_embeddings = df_additional_emb.drop(columns=[functionality_column]).values

            label_artifact = loaded_artifacts.get("df_celltypes")
            label_summary = None
            if label_artifact is not None:
                df_celltypes_emb = label_artifact["dataframe"]
                label_column = (
                    annotation_column
                    if annotation_column in df_celltypes_emb.columns
                    else "cell_type"
                )
                label_embeddings = df_celltypes_emb.drop(columns=[label_column]).values
                label_names = df_celltypes_emb[label_column].tolist()
                label_similarity = _similarity_matrix(
                    label_embeddings,
                    functionality_embeddings,
                    config.similarity_metric,
                )
                label_summary = pd.DataFrame(
                    [
                        {
                            "functionality": functionality_name,
                            "cell_type": cell_type,
                            "label_embedding_similarity": float(
                                label_similarity[label_idx, functionality_idx]
                            ),
                        }
                        for label_idx, cell_type in enumerate(label_names)
                        for functionality_idx, functionality_name in enumerate(functionality_names)
                    ]
                )

            print(functionality_names)

            run_dir = create_evaluation_run_directory(
                output_dir=config.output_dir,
                model_name=model_name,
                dataset_name=dataset_name,
                evaluation_name="functionality_similarity",
                timestamp=loaded_dataset["run_timestamp"],
            )

            # --- Evaluate similarity per functionality embedding ---
            results_df, _ = evaluate_similarity(
                cell_embeddings=cell_embeddings,
                other_embeddings=functionality_embeddings,
                other_labels=functionality_names,
                ground_truth=ground_truth,
                cell_umap=cell_umap,
                other_umap=None,
                similarity_metric=config.similarity_metric,
                output_dir=run_dir,
                bins=config.bins
            )

            # Compute mean AUC per functionality × cell type
            auc_summary = (
                results_df.groupby(["other_embedding", "ground_truth_column"], sort=False)["roc_auc"]
                .mean()
                .reset_index()
                .rename(columns={
                    "other_embedding": "functionality",
                    "ground_truth_column": "cell_type",
                    "roc_auc": "mean_auc"
                })
            )
            auc_summary["model"] = model_name
            auc_summary["dataset"] = dataset_name

            similarity_matrix = _similarity_matrix(
                cell_embeddings,
                functionality_embeddings,
                config.similarity_metric,
            )
            mean_cell_similarity_summary = pd.DataFrame(
                [
                    {
                        "functionality": functionality_name,
                        "cell_type": cell_type,
                        "mean_cell_similarity": float(
                            similarity_matrix[
                                (cell_annotations == cell_type).to_numpy(),
                                functionality_idx,
                            ].mean()
                        ),
                    }
                    for cell_type in cell_types
                    for functionality_idx, functionality_name in enumerate(functionality_names)
                ]
            )
            median_cell_similarity_summary = pd.DataFrame(
                [
                    {
                        "functionality": functionality_name,
                        "cell_type": cell_type,
                        "median_cell_similarity": float(
                            np.median(
                                similarity_matrix[
                                    (cell_annotations == cell_type).to_numpy(),
                                    functionality_idx,
                                ]
                            )
                        ),
                    }
                    for cell_type in cell_types
                    for functionality_idx, functionality_name in enumerate(functionality_names)
                ]
            )

            combined_summary = auc_summary.merge(
                mean_cell_similarity_summary,
                on=["functionality", "cell_type"],
                how="left",
            )
            combined_summary = combined_summary.merge(
                median_cell_similarity_summary,
                on=["functionality", "cell_type"],
                how="left",
            )
            if label_summary is not None:
                combined_summary = combined_summary.merge(
                    label_summary,
                    on=["functionality", "cell_type"],
                    how="left",
                )
            else:
                combined_summary["label_embedding_similarity"] = np.nan

            all_results.append(combined_summary)
            
            print(combined_summary.head())
            
            # Convert the column in the DataFrame itself
            combined_summary["functionality"] = pd.Categorical(
                combined_summary["functionality"],
                categories=functionality_names,
                ordered=True
            )

            roc_auc_heatmap_df = combined_summary.pivot(
                index="cell_type",
                columns="functionality",
                values="mean_auc"
            )
            
            roc_auc_heatmap_df = roc_auc_heatmap_df.sort_index()

            mean_cell_similarity_heatmap_df = combined_summary.pivot(
                index="cell_type",
                columns="functionality",
                values="mean_cell_similarity"
            ).sort_index()

            median_cell_similarity_heatmap_df = combined_summary.pivot(
                index="cell_type",
                columns="functionality",
                values="median_cell_similarity"
            ).sort_index()

            label_similarity_heatmap_df = combined_summary.pivot(
                index="cell_type",
                columns="functionality",
                values="label_embedding_similarity"
            ).sort_index()

            print(roc_auc_heatmap_df.head())

            colormap_name = "Heatmap: Teal–White–Red"
            plotter = UMAPCellPlotter(colormap_name=colormap_name)

            heatmap_paths = {}
            heatmap_paths["roc_auc"] = _plot_heatmap_outputs(
                plotter=plotter,
                heatmap_df=roc_auc_heatmap_df,
                run_dir=run_dir,
                file_stem="functionality_heatmap",
                title="Functionality ROC-AUC Heatmap",
                true_label_map=config.true_label_map,
            )
            heatmap_paths["roc_auc_named"] = _plot_heatmap_outputs(
                plotter=plotter,
                heatmap_df=roc_auc_heatmap_df,
                run_dir=run_dir,
                file_stem="functionality_roc_auc_heatmap",
                title="Functionality ROC-AUC Heatmap",
                true_label_map=config.true_label_map,
            )
            heatmap_paths["mean_cell_similarity"] = _plot_heatmap_outputs(
                plotter=plotter,
                heatmap_df=mean_cell_similarity_heatmap_df,
                run_dir=run_dir,
                file_stem="functionality_mean_cell_similarity_heatmap",
                title="Mean Cell-Level Similarity Heatmap",
                true_label_map=config.true_label_map,
            )
            heatmap_paths["median_cell_similarity"] = _plot_heatmap_outputs(
                plotter=plotter,
                heatmap_df=median_cell_similarity_heatmap_df,
                run_dir=run_dir,
                file_stem="functionality_median_cell_similarity_heatmap",
                title="Median Cell-Level Similarity Heatmap",
                true_label_map=config.true_label_map,
            )
            heatmap_paths["label_embedding_similarity"] = _plot_heatmap_outputs(
                plotter=plotter,
                heatmap_df=label_similarity_heatmap_df,
                run_dir=run_dir,
                file_stem="functionality_label_embedding_similarity_heatmap",
                title="Label-Embedding Similarity Heatmap",
                true_label_map=config.true_label_map,
            )

            results_path = run_dir / "results_df.csv"
            combined_summary.to_csv(results_path, index=False)
            auc_summary.to_csv(run_dir / "functionality_roc_auc_summary.csv", index=False)
            mean_cell_similarity_summary.to_csv(
                run_dir / "functionality_mean_cell_similarity_summary.csv",
                index=False,
            )
            median_cell_similarity_summary.to_csv(
                run_dir / "functionality_median_cell_similarity_summary.csv",
                index=False,
            )
            if label_summary is not None:
                label_summary.to_csv(
                    run_dir / "functionality_label_embedding_similarity_summary.csv",
                    index=False,
                )
            write_metadata(
                run_dir,
                {
                    "evaluation_name": "functionality_similarity",
                    "model_name": model_name,
                    "dataset_name": dataset_name,
                    "run_timestamp": run_dir.name,
                    "embedding_run_dir": cell_meta.get("run_dir"),
                    "cell_umap_path": cell_meta.get("umap", {}).get("path"),
                    "results_path": str(results_path),
                    "heatmap_path": str(run_dir / "functionality_heatmap.pdf"),
                    "heatmap_paths": heatmap_paths,
                    "n_rows": len(auc_summary),
                    "true_label_map": _json_safe_true_label_map(config.true_label_map),
                },
            )

    combined_results = pd.concat(all_results, ignore_index=True)

    
    return combined_results


def _json_safe_true_label_map(
    true_label_map: dict[str, Any] | None,
) -> dict[str, list[str]] | None:
    if true_label_map is None:
        return None
    clean = {}
    for functionality, labels in true_label_map.items():
        if isinstance(labels, str):
            labels = [labels]
        clean[str(functionality)] = [str(label) for label in labels]
    return clean
