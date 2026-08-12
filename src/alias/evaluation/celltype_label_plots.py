from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Mapping
import pandas as pd
import numpy as np
import json

from alias.util.artifacts import create_evaluation_run_directory, write_metadata

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
    from sklearn.decomposition import PCA
    import umap

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


def get_umap_plotter_class():
    from alias.util.plots.umap_plots import UMAPCellPlotter

    return UMAPCellPlotter


def _numeric_embedding_array(df: pd.DataFrame) -> np.ndarray:
    embedding_columns = [
        column
        for column in df.columns
        if isinstance(column, (int, np.integer))
        or (isinstance(column, str) and column.isdigit())
    ]
    if not embedding_columns:
        raise ValueError("No numeric embedding coordinate columns found for UMAP computation.")
    return df.loc[:, embedding_columns].to_numpy(dtype=np.float32)


def _plot_grouped_umaps(
    df_cells_emb: pd.DataFrame,
    df_celltypes_emb: pd.DataFrame | None,
    annotation_column: str,
    celltype_groups: Mapping[str, set[str] | list[str] | tuple[str, ...]],
    figure_run_dir: Path,
    derived_umap_dir: Path,
    evaluation_config: EvaluationConfig,
    time_column: str = "time",
) -> dict[str, dict[str, Any]]:
    group_artifacts: dict[str, dict[str, Any]] = {}
    plotter_class = get_umap_plotter_class()

    for group_name, group_set in celltype_groups.items():
        group_labels = {str(label) for label in group_set}
        df_cells_group = df_cells_emb[
            df_cells_emb[annotation_column].astype(str).isin(group_labels)
        ].copy()
        if df_cells_group.empty:
            continue

        df_centroids_group = None
        if df_celltypes_emb is not None and annotation_column in df_celltypes_emb.columns:
            group_celltypes = df_cells_group[annotation_column].astype(str).unique().tolist()
            df_centroids_group = df_celltypes_emb[
                df_celltypes_emb[annotation_column].astype(str).isin(group_celltypes)
            ].copy()

        dfs_to_concat = [df_cells_group]
        if df_centroids_group is not None and not df_centroids_group.empty:
            df_centroids_group["batch"] = "celltype"
            dfs_to_concat.append(df_centroids_group)

        df_cells_group["batch"] = "cell"
        df_combined = pd.concat(dfs_to_concat, axis=0, ignore_index=False, join="outer")
        embeddings_array = _numeric_embedding_array(df_combined)
        print(f"Computing grouped UMAP for {group_name} ({len(df_combined)} embeddings)...")
        umap_coords = compute_umap(embeddings_array, evaluation_config)
        df_combined[["UMAP1", "UMAP2"]] = umap_coords

        df_cells_umap = df_combined[df_combined["batch"] == "cell"].copy()
        if time_column in df_cells_umap.columns:
            df_cells_umap[time_column] = pd.to_numeric(df_cells_umap[time_column], errors="coerce")
        df_centroids_umap = None
        if df_centroids_group is not None and not df_centroids_group.empty:
            df_centroids_umap = df_combined.loc[
                df_combined["batch"] == "celltype",
                [annotation_column, "UMAP1", "UMAP2"],
            ].copy()
            df_centroids_umap = (
                df_centroids_umap.rename(columns={annotation_column: "cell_type"})
                .reset_index(drop=True)
            )

        group_fig_dir = figure_run_dir / str(group_name)
        group_fig_dir.mkdir(parents=True, exist_ok=True)
        group_umap_dir = derived_umap_dir / str(group_name)
        group_umap_dir.mkdir(parents=True, exist_ok=True)

        umap_cells_path = group_umap_dir / "df_cells_umap.parquet"
        df_cells_umap.to_parquet(umap_cells_path, index=True)
        centroids_path = None
        if df_centroids_umap is not None:
            centroids_path = group_umap_dir / "df_celltypes_umap.parquet"
            df_centroids_umap.to_parquet(centroids_path, index=True)

        plotter = plotter_class()
        plotter.annotate_centroids = False
        plotter.plot_cells(
            df_cells_umap,
            annotation_column=annotation_column,
            output_path=group_fig_dir / f"cells_colored_by_annotation_{group_name}.pdf",
            title=f"Cells Colored by Annotation - {group_name}",
        )
        if time_column in df_cells_umap.columns:
            plotter.plot_cells(
                df_cells_umap,
                time_color_column=time_column,
                output_path=group_fig_dir / f"cells_colored_by_time_{group_name}.pdf",
                title=f"Cells Colored by Time - {group_name}",
            )

        if df_centroids_umap is not None:
            plotter.annotate_centroids = True
            labeled_annotation_stem = group_fig_dir / f"cells_colored_by_annotation_with_labels_{group_name}"
            plotter.plot_cells(
                df_cells_umap,
                annotation_column=annotation_column,
                annotate_centroids_df=df_centroids_umap,
                output_path=labeled_annotation_stem.with_suffix(".pdf"),
                title=f"Cells with Cell Type Labels - {group_name}",
            )
            plotter.plot_cells(
                df_cells_umap,
                annotation_column=annotation_column,
                annotate_centroids_df=df_centroids_umap,
                output_path=labeled_annotation_stem.with_suffix(".svg"),
                title=f"Cells with Cell Type Labels - {group_name}",
            )
            if time_column in df_cells_umap.columns:
                labeled_time_stem = group_fig_dir / f"cells_colored_by_time_with_labels_{group_name}"
                plotter.plot_cells(
                    df_cells_umap,
                    time_color_column=time_column,
                    annotate_centroids_df=df_centroids_umap,
                    output_path=labeled_time_stem.with_suffix(".pdf"),
                    title=f"Cells with Cell Type Labels - {group_name}",
                )
                plotter.plot_cells(
                    df_cells_umap,
                    time_color_column=time_column,
                    annotate_centroids_df=df_centroids_umap,
                    output_path=labeled_time_stem.with_suffix(".svg"),
                    title=f"Cells with Cell Type Labels - {group_name}",
                )

        group_artifacts[str(group_name)] = {
            "df_cells_umap": {"path": str(umap_cells_path), "n_points": len(df_cells_umap)},
            "df_celltypes_umap": {
                "path": str(centroids_path),
                "n_points": len(df_centroids_umap) if df_centroids_umap is not None else 0,
            }
            if centroids_path is not None
            else None,
            "cell_types": sorted(group_labels),
        }

    return group_artifacts

def umap_plots(
    embeddings_dict: Dict[str, Dict[str, Dict[str, Any]]],
    annotation_column: str,
    output_dir: str,
    evaluation_config: EvaluationConfig,
    extra_cell_annotations: Mapping[str, Mapping[str, Any] | pd.Series] | None = None,
    celltype_groups: Mapping[str, set[str] | list[str] | tuple[str, ...]] | None = None,
    time_column: str = "time",
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Generate UMAP plots for all models and datasets, save UMAP coordinates
    next to embeddings, and update embeddings_dict with UMAP paths.
    """
    figure_root = Path(output_dir)

    for model_name, model_data in embeddings_dict.items():
        print(f"Evaluating model: {model_name}")

        for dataset_name, dataset_meta in model_data.items():
            print(f"Processing dataset: {dataset_name}")

            # --- Load cell embeddings ---
            cell_info = dataset_meta.get("df_cells")
            if cell_info is None or "path" not in cell_info:
                print(f"⚠️ Skipping dataset {dataset_name}: no cell embeddings found.")
                continue

            embedding_run_dir = Path(cell_info["path"]).parent
            run_timestamp = embedding_run_dir.name
            figure_run_dir = create_evaluation_run_directory(
                output_dir=figure_root,
                model_name=model_name,
                dataset_name=dataset_name,
                evaluation_name="celltype_label_plots",
                timestamp=run_timestamp,
            )
            derived_umap_dir = embedding_run_dir / "umap" / "celltype_label_plots" / figure_run_dir.name
            derived_umap_dir.mkdir(parents=True, exist_ok=True)

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

            extra_cell_annotations = extra_cell_annotations or {}
            for extra_column, extra_values in extra_cell_annotations.items():
                if extra_column in df_cells_emb.columns:
                    continue
                if isinstance(extra_values, pd.Series):
                    value_map = extra_values.astype(str).to_dict()
                else:
                    value_map = {str(key): value for key, value in dict(extra_values).items()}
                df_cells_emb[extra_column] = df_cells_emb.index.map(
                    lambda idx: value_map.get(str(idx), "unknown")
                )

            if time_column in df_cells_emb.columns:
                df_cells_emb[time_column] = pd.to_numeric(df_cells_emb[time_column], errors="coerce")

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

            if celltype_groups is not None:
                group_artifacts = _plot_grouped_umaps(
                    df_cells_emb=df_cells_emb,
                    df_celltypes_emb=df_celltypes_emb,
                    annotation_column=annotation_column,
                    celltype_groups=celltype_groups,
                    figure_run_dir=figure_run_dir,
                    derived_umap_dir=derived_umap_dir,
                    evaluation_config=evaluation_config,
                    time_column=time_column,
                )
                embeddings_dict[model_name][dataset_name]["celltype_group_umaps"] = group_artifacts
                write_metadata(
                    figure_run_dir,
                    {
                        "dataset_name": dataset_name,
                        "model_name": model_name,
                        "evaluation_name": "celltype_label_plots",
                        "run_timestamp": figure_run_dir.name,
                        "embedding_run_dir": str(embedding_run_dir),
                        "umap_run_dir": str(derived_umap_dir),
                        "grouped_mode": True,
                        "celltype_groups": {
                            str(key): sorted(map(str, values))
                            for key, values in celltype_groups.items()
                        },
                        "artifacts": group_artifacts,
                        "extra_cell_annotations": list(extra_cell_annotations.keys()),
                    },
                )
                print(f"✅ Saved grouped UMAP coords and plots for {model_name} / {dataset_name}\n")
                continue

            # --- Combine for joint UMAP ---
            dfs_to_concat = [df_cells_emb]
            if df_celltypes_emb is not None:
                dfs_to_concat.append(df_celltypes_emb)
                df_celltypes_emb["batch"] = "celltype"

            df_cells_emb["batch"] = "cell"
            df_combined = pd.concat(
                dfs_to_concat,
                axis=0,
                ignore_index=False,
                join="outer"         # ensures all columns are included
            )
            
            embeddings_array = _numeric_embedding_array(df_combined)
            
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
            umap_cells_path = derived_umap_dir / "df_cells_umap.parquet"
            df_cells_umap.to_parquet(umap_cells_path, index=True)
            embeddings_dict[model_name][dataset_name]["df_cells"]["umap"] = {
                "path": str(umap_cells_path),
                "n_points": len(df_cells_umap)
            }

            if df_centroids_umap is not None:
                umap_centroids_path = derived_umap_dir / "df_celltypes_umap.parquet"
                df_centroids_umap.to_parquet(umap_centroids_path, index=True)
                embeddings_dict[model_name][dataset_name]["df_celltypes"]["umap"] = {
                    "path": str(umap_centroids_path),
                    "n_points": len(df_centroids_umap)
                }

            # --- Plotting ---
            plotter = get_umap_plotter_class()()

            # Cells colored by annotation
            plotter.annotate_centroids = False
            plotter.plot_cells(
                df_cells_umap,
                annotation_column=annotation_column,
                output_path=figure_run_dir / "cells_colored_by_annotation.svg",
                title="Cells Colored by Annotation",
            )
            for extra_column in extra_cell_annotations:
                plotter.plot_cells(
                    df_cells_umap,
                    annotation_column=extra_column,
                    output_path=figure_run_dir / f"cells_colored_by_{extra_column.replace('.', '_')}.svg",
                    title=f"Cells Colored by {extra_column}",
                )

            # Cells with centroids (cell type labels)
            if df_centroids_umap is not None:
                plotter.annotate_centroids = True
                plotter.plot_cells(
                    df_cells_umap,
                    annotation_column=annotation_column,
                    annotate_centroids_df=df_centroids_umap,
                    output_path=figure_run_dir / "cells_with_celltype_labels.svg",
                    title="Cells with Cell Type Labels",
                )

            write_metadata(
                figure_run_dir,
                {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "evaluation_name": "celltype_label_plots",
                    "run_timestamp": figure_run_dir.name,
                    "embedding_run_dir": str(embedding_run_dir),
                    "umap_run_dir": str(derived_umap_dir),
                    "artifacts": {
                        "df_cells_umap": embeddings_dict[model_name][dataset_name]["df_cells"]["umap"],
                        "df_celltypes_umap": embeddings_dict[model_name][dataset_name]
                        .get("df_celltypes", {})
                        .get("umap"),
                    },
                    "extra_cell_annotations": list(extra_cell_annotations.keys()),
                },
            )

            print(f"✅ Saved UMAP coords and plots for {model_name} / {dataset_name}\n")

    return embeddings_dict
