import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from alias.util.similarity import evaluate_similarity
from alias.util.plots.umap_plots import UMAPCellPlotter
import json


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

            # --- Load cell embeddings ---
            cell_meta = dataset_meta["df_cells"]
            cell_df = pd.read_parquet(cell_meta["path"])
            cell_df.index = cell_df.index.astype(str)

            # Load annotations from JSON if present
            ann_path = cell_meta.get("annotation_map")
            if ann_path and Path(ann_path).exists():
                with open(ann_path, "r") as f:
                    annotation_map_full = json.load(f)
                # Extract the dict for the specific annotation column
                annotation_map = annotation_map_full.get(annotation_column, {})
                cell_df[annotation_column] = cell_df.index.map(
                    lambda idx: annotation_map.get(idx, "unknown")
                )
            elif annotation_column not in cell_df.columns:
                cell_df[annotation_column] = "unknown"

            cell_embeddings = cell_df.drop(columns=[annotation_column]).values
            cell_annotations = cell_df[annotation_column]

            # --- Load cell type embeddings ---
            label_meta = dataset_meta.get("df_celltypes")
            if label_meta is None:
                continue

            df_celltypes_emb = pd.read_parquet(label_meta["path"])
            df_celltypes_emb.index = df_celltypes_emb.index.astype(str)

            # Load annotations for cell types
            ann_path = label_meta.get("annotation_map")
            if ann_path and Path(ann_path).exists():
                with open(ann_path, "r") as f:
                    annotation_map = json.load(f)
                df_celltypes_emb[annotation_column] = df_celltypes_emb.index.map(
                    lambda idx: annotation_map.get(idx, "unknown")  # <- direct value
                )

            label_embeddings = df_celltypes_emb.drop(columns=[annotation_column]).values
            cell_type_labels = df_celltypes_emb[annotation_column].tolist()

            # --- Load optional UMAPs ---
            cell_umap_dict = {}
            cell_type_umap_dict = {}
            for key, val in dataset_meta.items():
                if isinstance(val, dict) and "umap" in val:
                    if key == "df_cells":
                        cell_umap_dict[key] = pd.read_parquet(val["umap"]["path"])
                    elif key == "df_celltypes":
                        cell_type_umap_dict[key] = pd.read_parquet(val["umap"]["path"])

            # --- Prepare output directory ---
            base_out = Path(config.output_dir) / model_name / dataset_name / "celltype_label_similarity"
            base_out.mkdir(parents=True, exist_ok=True)

            # --- Evaluate similarity per cell type ---
            for i, cell_type in enumerate(cell_type_labels):
                ground_truth = pd.DataFrame({cell_type: cell_annotations == cell_type})

                other_embedding = label_embeddings[i].reshape(1, -1)
                other_label = [cell_type]

                # Select UMAPs if available
                cell_umap = next(iter(cell_umap_dict.values()), None)
                cell_type_umap = None
                if cell_type_umap_dict:
                    cell_type_umap = next(iter(cell_type_umap_dict.values())).iloc[[i]]

                results_df, _ = evaluate_similarity(
                    cell_embeddings=cell_embeddings,
                    other_embeddings=other_embedding,
                    other_labels=other_label,
                    ground_truth=ground_truth,
                    cell_umap=cell_umap,
                    other_umap=cell_type_umap,
                    similarity_metric=config.similarity_metric,
                    output_dir=base_out,
                    bins=config.bins
                )

                # Add metadata
                results_df["model_name"] = model_name
                results_df["dataset_name"] = dataset_name
                results_df["cell_type"] = cell_type

                all_results.append(results_df)

    combined_results = pd.concat(all_results, ignore_index=True)
    return combined_results
