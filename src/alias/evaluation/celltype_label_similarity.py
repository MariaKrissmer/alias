import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from alias.util.similarity import evaluate_similarity

from alias.util.artifacts import create_evaluation_run_directory, write_metadata
from alias.evaluation.embedding import load_dataset_embedding_artifacts


@dataclass
class CellTypeSimilarityConfig:
    similarity_metric: str = "cosine"
    bins: int = 60
    output_dir: Path = Path(".")  # base folder for evaluation_plots

def cell_type_label_similarity(
    embeddings_dict: dict,
    annotation_column: str,
    config: CellTypeSimilarityConfig
) -> pd.DataFrame:
    """
    Compute similarity between cell embeddings and cell type label embeddings
    across all models and datasets in embeddings_dict.
    Handles loading annotations from JSON and optional UMAP coordinates.
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

            cell_embeddings = cell_df.drop(columns=[annotation_column]).values
            cell_annotations = cell_df[annotation_column]

            label_artifact = loaded_artifacts.get("df_celltypes")
            if label_artifact is None:
                continue

            run_dir = create_evaluation_run_directory(
                output_dir=config.output_dir,
                model_name=model_name,
                dataset_name=dataset_name,
                evaluation_name="celltype_label_similarity",
                timestamp=loaded_dataset["run_timestamp"],
            )
            dataset_results = []

            label_meta = label_artifact["metadata"]
            df_celltypes_emb = label_artifact["dataframe"]
            label_column = annotation_column if annotation_column in df_celltypes_emb.columns else "cell_type"

            label_embeddings = df_celltypes_emb.drop(columns=[label_column]).values
            cell_type_labels = df_celltypes_emb[label_column].tolist()
            cell_umap = cell_artifact["umap"]
            cell_type_umap = label_artifact["umap"]

            # --- Evaluate similarity per cell type ---
            for i, cell_type in enumerate(cell_type_labels):
                ground_truth = pd.DataFrame({cell_type: cell_annotations == cell_type})

                other_embedding = label_embeddings[i].reshape(1, -1)
                other_label = [cell_type]

                selected_cell_type_umap = None
                if cell_type_umap is not None:
                    selected_cell_type_umap = cell_type_umap.iloc[[i]]

                results_df, _ = evaluate_similarity(
                    cell_embeddings=cell_embeddings,
                    other_embeddings=other_embedding,
                    other_labels=other_label,
                    ground_truth=ground_truth,
                    cell_umap=cell_umap,
                    other_umap=selected_cell_type_umap,
                    similarity_metric=config.similarity_metric,
                    output_dir=run_dir,
                    bins=config.bins
                )

                # Add metadata
                results_df["model_name"] = model_name
                results_df["dataset_name"] = dataset_name
                results_df["cell_type"] = cell_type

                all_results.append(results_df)
                dataset_results.append(results_df)

            if dataset_results:
                combined_dataset_results = pd.concat(dataset_results, ignore_index=True)
                results_path = run_dir / "results_df.csv"
                combined_dataset_results.to_csv(results_path, index=False)
                write_metadata(
                    run_dir,
                    {
                        "evaluation_name": "celltype_label_similarity",
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "run_timestamp": run_dir.name,
                        "embedding_run_dir": cell_meta.get("run_dir"),
                        "cell_umap_path": cell_meta.get("umap", {}).get("path"),
                        "celltype_umap_path": dataset_meta.get("df_celltypes", {}).get("umap", {}).get("path"),
                        "results_path": str(results_path),
                        "n_rows": len(combined_dataset_results),
                    },
                )

    combined_results = pd.concat(all_results, ignore_index=True)
    return combined_results
