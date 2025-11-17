import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.metrics import roc_curve, auc

from util.plots.umap_plots import UMAPCellPlotter


def evaluate_similarity_cellwhisperer(
    sim_matrix: pd.DataFrame | np.ndarray,
    other_labels: list[str] | np.ndarray | None = None,
    ground_truth: pd.DataFrame | None = None,
    cell_umap: pd.DataFrame | None = None,
    other_umap: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    bins: int = 60
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Evaluate similarity performance when the similarity matrix is already precomputed.
    The sim_matrix should be shape [n_cells × n_functionalities].
    """

    if isinstance(sim_matrix, pd.DataFrame):
        if other_labels is None:
            other_labels = list(sim_matrix.columns)
        sim_matrix = sim_matrix.to_numpy()
    elif other_labels is None:
        raise ValueError("Must provide `other_labels` if sim_matrix is a numpy array.")

    plotter = UMAPCellPlotter()
    if output_dir is not None:
        output_dir = Path(output_dir)
        roc_dir = output_dir / "roc_curves"
        umap_dir = output_dir / "umap"
        hist_dir = output_dir / "histograms"
        for d in [roc_dir, umap_dir, hist_dir]:
            d.mkdir(parents=True, exist_ok=True)

    results = []

    # --- Main evaluation loop ---
    for i, other_label in enumerate(other_labels):
        sim_scores = sim_matrix[:, i]

        if ground_truth is not None:
            for gt_col in ground_truth.columns:
                y_true = ground_truth[gt_col].astype(int).values

                # Compute ROC + AUC
                fpr, tpr, thresholds = roc_curve(y_true, sim_scores)
                roc_auc = auc(fpr, tpr)

                results.append({
                    "other_embedding": other_label,
                    "ground_truth_column": gt_col,
                    "roc_auc": roc_auc,
                    "fpr": fpr,
                    "tpr": tpr,
                    "thresholds": thresholds
                })

                if output_dir is not None:
                    safe_label = re.sub(r"[^\w\d\-\.]", "_", str(other_label))
                    df_roc = pd.DataFrame({"fpr": fpr, "tpr": tpr, "auc": roc_auc})
                    plotter.plot_roc(
                        df_roc,
                        output_path=(roc_dir / f"{safe_label}_{gt_col}_roc.pdf"),
                        title=f"{other_label} vs {gt_col}"
                    )

                    df_hist = pd.DataFrame({
                        "similarity": sim_scores,
                        "group": np.where(y_true, str(gt_col), "other")
                    })
                    plotter.plot_similarity_histogram(
                        df=df_hist,
                        label=str(gt_col),
                        output_path=hist_dir / f"{safe_label}_{gt_col}_hist.pdf",
                        bins=bins
                    )

        # --- Optional UMAP overlay ---
        if cell_umap is not None:
            df_umap_plot = pd.DataFrame({
                "UMAP1": cell_umap["UMAP1"],
                "UMAP2": cell_umap["UMAP2"],
                "Similarity Score": sim_scores
            })
            df_other_umap = other_umap.iloc[[i]] if other_umap is not None else None
            if output_dir is not None:
                plotter.annotate_centroids = df_other_umap is not None
                plotter.plot_cells(
                    df=df_umap_plot,
                    continuous_color_column="Similarity Score",
                    annotate_centroids_df=df_other_umap,
                    output_path=umap_dir / f"{other_label}_umap.pdf",
                    title=str(other_label)
                )

    results_df = pd.DataFrame(results)
    return results_df, sim_matrix
