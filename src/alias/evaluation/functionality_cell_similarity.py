import pandas as pd
from pathlib import Path
from dataclasses import dataclass
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

            all_results.append(auc_summary)
            
            print(auc_summary.head())
            
            # Convert the column in the DataFrame itself
            auc_summary["functionality"] = pd.Categorical(
                auc_summary["functionality"],
                categories=functionality_names,
                ordered=True
            )

            heatmap_df = auc_summary.pivot(
                index="cell_type",
                columns="functionality",
                values="mean_auc"
            )
            
            heatmap_df = heatmap_df.sort_index()

            print(heatmap_df.head())

            # Plot
            colormap_name = "Heatmap: Teal–White–Red"
            plotter = UMAPCellPlotter(colormap_name=colormap_name)
            plotter.plot_similarity_heatmap(
                sim_df=heatmap_df, 
                output_path=run_dir / "functionality_heatmap.pdf"
            )

            results_path = run_dir / "results_df.csv"
            auc_summary.to_csv(results_path, index=False)
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
                    "n_rows": len(auc_summary),
                },
            )

    combined_results = pd.concat(all_results, ignore_index=True)

    
    return combined_results
