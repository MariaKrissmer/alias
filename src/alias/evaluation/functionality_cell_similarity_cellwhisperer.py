import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from alias.util.similarity_cellwhisperer import evaluate_similarity_cellwhisperer
import json

from alias.util.plots.umap_plots import UMAPCellPlotter

@dataclass
class FunctionalitySimilarityConfig:
    similarity_metric: str = "cosine"
    bins: int = 60
    output_dir: Path = Path(".")
    plot: bool = True

def functionality_similarity_cellwhisperer(
    similarity_df: pd.DataFrame,
    cell_annotations: pd.Series,
    functionality_names: list[str],
    config: FunctionalitySimilarityConfig,
    cell_umap: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Evaluate similarity results when a precomputed similarity matrix is provided.
    `similarity_df` should have shape [n_cells × n_functionalities].
    """

    print("Evaluating precomputed similarity matrix...")

    cell_types = sorted(cell_annotations.unique())
    ground_truth = pd.DataFrame({ct: cell_annotations == ct for ct in cell_types})

    # Output directory
    base_out = Path(config.output_dir) 
    
    results_df, _ = evaluate_similarity_cellwhisperer(
        sim_matrix=similarity_df,
        other_labels=functionality_names,
        ground_truth=ground_truth,
        cell_umap=cell_umap,
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

    # Plot heatmap
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
    
    from util.plots.umap_plots import UMAPCellPlotter
    plotter = UMAPCellPlotter(colormap_name="Heatmap: Teal–White–Red")
    plotter.plot_similarity_heatmap(
        sim_df=heatmap_df,
        output_path=base_out / "functionality_heatmap.pdf"
    )

    return auc_summary
