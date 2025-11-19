from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import umap
from alias.util.plots.umap_plots import UMAPCellPlotter
import json

@dataclass
class EvaluationConfig:
    n_neighbors: int
    min_dist: float = 0.5
    resolution: float = 0.5
    n_components: int = 50
    random_state: int = 73

def compute_umap(embeddings: np.ndarray, evaluation_config: EvaluationConfig, n_pca: int = 50):
    """
    Compute PCA followed by UMAP for dimensionality reduction.
    """
    if embeddings.shape[0] == 0:
        raise ValueError("No embeddings provided for UMAP computation.")
    
    # Clean bad values before PCA
    if not np.isfinite(embeddings).all():
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    pca_model = PCA(n_components=evaluation_config.n_components,
                    random_state=evaluation_config.random_state,
                    svd_solver='randomized') 
    embeddings_pca = pca_model.fit_transform(embeddings)

    umap_model = umap.UMAP(
        n_components=2,
        n_neighbors=evaluation_config.n_neighbors,
        min_dist=evaluation_config.min_dist,
        random_state=evaluation_config.random_state,
    )
    return umap_model.fit_transform(embeddings_pca)

def umap_plots(
    embeddings_dict: Dict[str, Dict[str, Dict[str, Any]]],
    annotation_column: str,
    output_dir: str,
    evaluation_config: EvaluationConfig
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Generate UMAP plots for all models and datasets, save UMAP coordinates
    next to embeddings, and update embeddings_dict with UMAP paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model_data in embeddings_dict.items():
        print(f"Evaluating model: {model_name}")
        model_dir = output_dir / model_name
        model_dir.mkdir(exist_ok=True)

        for dataset_name, dataset_meta in model_data.items():
            print(f"Processing dataset: {dataset_name}")
            dataset_dir = model_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)

            # --- Load cell embeddings ---
            cell_info = dataset_meta.get("df_cells")
            if cell_info is None or "path" not in cell_info:
                print(f"⚠️ Skipping dataset {dataset_name}: no cell embeddings found.")
                continue

            df_cells_emb = pd.read_parquet(cell_info["path"])
            df_cells_emb.index = df_cells_emb.index.astype(str)

            # Restore annotation from JSON if available
            ann_path = cell_info.get("annotation_map")
            if ann_path and Path(ann_path).exists():
                with open(ann_path, "r") as f:
                    annotation_map_full = json.load(f)
                # Get the dict for this specific annotation column
                annotation_map = annotation_map_full.get(annotation_column, {})
                # Map cell indices to their annotation values
                df_cells_emb[annotation_column] = df_cells_emb.index.map(
                    lambda idx: annotation_map.get(idx, "unknown")
                )
            elif annotation_column not in df_cells_emb.columns:
                df_cells_emb[annotation_column] = "unknown"


            # --- Load celltype embeddings (optional) ---
            df_celltypes_emb = None
            if "df_celltypes" in dataset_meta:
                ct_info = dataset_meta["df_celltypes"]
                df_celltypes_emb = pd.read_parquet(ct_info["path"])
                df_celltypes_emb.index = df_celltypes_emb.index.astype(str)

                # Restore annotation from annotation_map JSON if available
                ann_path = ct_info.get("annotation_map")
                if ann_path and Path(ann_path).exists():
                    with open(ann_path, "r") as f:
                        annotation_map = json.load(f)
                    df_celltypes_emb[annotation_column] = df_celltypes_emb.index.map(
                        lambda idx: annotation_map.get(idx, "unknown")  # <- direct value
                    )
            
            print(df_celltypes_emb.head(10))

            # --- Combine for joint UMAP ---
            dfs_to_concat = [df_cells_emb]
            if df_celltypes_emb is not None:
                dfs_to_concat.append(df_celltypes_emb)
                df_celltypes_emb["batch"] = "celltype"

            df_cells_emb["batch"] = "cell"
            df_combined = pd.concat(
                [df_cells_emb, df_celltypes_emb],
                axis=0,
                ignore_index=True,   # ignores existing indices
                join="outer"         # ensures all columns are included
            )
            
            embeddings_array = df_combined.select_dtypes(include=[np.number]).to_numpy()
            
            print(f"Computing UMAP for {len(df_combined)} embeddings...")
            umap_coords = compute_umap(embeddings_array, evaluation_config)

            # Add UMAP coordinates
            df_combined[["UMAP1", "UMAP2"]] = umap_coords

            # Split back
            df_cells_umap = df_combined[df_combined["batch"] == "cell"].copy()

            df_centroids_umap = None
            if df_celltypes_emb is not None:
                df_centroids_umap = df_combined.loc[
                    df_combined["batch"] == "celltype",  # filter rows
                    [annotation_column, "UMAP1", "UMAP2"]  # select only the columns you need
                ].copy()

                # Rename the annotation column to 'cell_type'
                df_centroids_umap = df_centroids_umap.rename(columns={annotation_column: "cell_type"}).reset_index(drop=True)
               
            # --- Save UMAP coordinates ---
            emb_dir = Path(cell_info["path"]).parent
            umap_cells_path = emb_dir / "df_cells_umap.parquet"
            df_cells_umap.to_parquet(umap_cells_path, index=True)
            embeddings_dict[model_name][dataset_name]["df_cells"]["umap"] = {
                "path": str(umap_cells_path),
                "n_points": len(df_cells_umap)
            }

            if df_centroids_umap is not None:
                umap_centroids_path = emb_dir / "df_celltypes_umap.parquet"
                df_centroids_umap.to_parquet(umap_centroids_path, index=True)
                embeddings_dict[model_name][dataset_name]["df_celltypes"]["umap"] = {
                    "path": str(umap_centroids_path),
                    "n_points": len(df_centroids_umap)
                }

            # --- Plotting ---
            plotter = UMAPCellPlotter()

            # Cells colored by annotation
            plotter.annotate_centroids = False
            plotter.plot_cells(
                df_cells_umap,
                annotation_column=annotation_column,
                output_path=dataset_dir / "cells_colored_by_annotation.pdf",
                title="Cells Colored by Annotation",
            )

            # Cells with centroids (cell type labels)
            if df_centroids_umap is not None:
                plotter.annotate_centroids = True
                plotter.plot_cells(
                    df_cells_umap,
                    annotation_column=annotation_column,
                    annotate_centroids_df=df_centroids_umap,
                    output_path=dataset_dir / "cells_with_celltype_labels.pdf",
                    title="Cells with Cell Type Labels",
                )

            print(f"✅ Saved UMAP coords and plots for {model_name} / {dataset_name}\n")

    return embeddings_dict
