import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from alias.util.similarity import evaluate_similarity
import json

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

            if annotation_column not in cell_df.columns:
                raise ValueError(f"{annotation_column} not found in cell dataframe for {dataset_name}")

            cell_embeddings = cell_df.drop(columns=[annotation_column]).values
            cell_annotations = cell_df[annotation_column]
            cell_types = sorted(cell_annotations.unique())
            
            ground_truth = pd.DataFrame({ct: cell_annotations == ct for ct in cell_types})

            # Optional cell UMAP
            cell_umap = None
            if "umap" in cell_meta and "path" in cell_meta["umap"]:
                cell_umap = pd.read_parquet(cell_meta["umap"]["path"])

            # --- Load functionality embeddings from df_additional ---
            additional_meta = dataset_meta.get("df_additional")
            if additional_meta is None:
                print(f"No df_additional found for {dataset_name}, skipping")
                continue

            df_additional_emb = pd.read_parquet(additional_meta["path"])
            df_additional_emb.index = df_additional_emb.index.astype(str)

            # Load mapping for descriptions
            mapping_path = additional_meta.get("annotation_map")
            if mapping_path and Path(mapping_path).exists():
                with open(mapping_path, "r") as f:
                    additional_mapping = json.load(f)
            else:
                additional_mapping = {idx: idx for idx in df_additional_emb.index}

            functionality_names = [additional_mapping[idx] for idx in df_additional_emb.index]
            functionality_embeddings = df_additional_emb.values

            print(functionality_names)
            
            # Prepare output folder
            base_out = Path(config.output_dir) / model_name / dataset_name / "functionality_similarity"
            base_out.mkdir(parents=True, exist_ok=True)

            # --- Evaluate similarity per functionality embedding ---
            results_df, _ = evaluate_similarity(
                cell_embeddings=cell_embeddings,
                other_embeddings=functionality_embeddings,
                other_labels=functionality_names,
                ground_truth=ground_truth,
                cell_umap=cell_umap,
                other_umap=None,
                similarity_metric=config.similarity_metric,
                output_dir=base_out,
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
                output_path=base_out / "functionality_heatmap.pdf"
            )

    combined_results = pd.concat(all_results, ignore_index=True)

    
    return combined_results

