from __future__ import annotations

import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _plotting_common import (  # noqa: E402
    AXIS_LABEL_SIZE,
    ModelSpec,
    PLOTS_DIR,
    PLOT_HEIGHT,
    TICK_LABEL_SIZE,
    concat_available,
    load_label_similarity,
    load_metrics_summary,
    ordered_blue_model_palette,
    publication_model_label,
    publication_model_labels,
    save_figure,
    set_plot_style,
)


MODELS = [
    ModelSpec("MB", "MB", ("MB_HIAI_Tcells_S2_N1_200k_lr5e5_e15", "MB_HIAI_Tcells_S2_N1_200k_lr5e5_e15_epoch_15_ncbi")),
    ModelSpec("MJ", "MJ", ("MJ_HIAI_Tcells_S2_N3_200k_lr5e5",)),
    ModelSpec("MG", "MG", ("MG_HIAI_Tcells_S2_200k_lr5e5", "MG_HIAI_Tcells_S2_200k")),
    ModelSpec("MF", "MF", ("MF_HIAI_Tcells_S3_200k_lr5e5",)),
    ModelSpec("MH", "MH", ("MH_HIAI_Tcells_S5_200k_lr5e5",)),
    ModelSpec("MI", "MI", ("MI_HIAI_Tcells_N1_200k_lr5e5",)),
    ModelSpec("Base", "PubMedBERTBase"),
]
OUTPUT_DIR = PLOTS_DIR / "ablation_annotation_effect"
MODEL_LEVEL_METRICS = [
    ("balanced_accuracy", "Balanced Accuracy"),
]
LABEL_OFFSETS = {
    "MB": (-22, 7),
    "MJ": (5, 5),
    "MG": (5, -10),
    "MF": (5, 9),
    "MH": (5, 5),
    "MI": (4, -12),
    "Base": (-18, 8),
}


def _select_models(model_names: list[str] | None = None) -> list[ModelSpec]:
    if not model_names:
        return MODELS
    requested = set(model_names)
    selected = [model for model in MODELS if model.label in requested]
    if not selected:
        raise ValueError(f"No ablation annotation plot models selected from: {model_names}")
    return selected


def load_ablation_data(models: list[ModelSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    roc_auc_df = concat_available(
        [load_label_similarity(model) for model in models],
        description="ablation ROC-AUC values",
    )
    metrics_df = concat_available(
        [load_metrics_summary(model) for model in models],
        description="ablation model-level annotation metrics",
    )
    mean_roc_auc_df = (
        roc_auc_df.groupby(["plot_model", "artifact_name"], as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            sd_roc_auc=("roc_auc", "std"),
            n_cell_types=("cell_type", "nunique"),
        )
    )
    model_summary = mean_roc_auc_df.merge(
        metrics_df,
        on=["plot_model", "artifact_name"],
        how="inner",
        validate="one_to_one",
    )
    if model_summary.empty:
        raise ValueError("No shared models between ROC-AUC summaries and annotation metrics.")
    return roc_auc_df, model_summary


def plot_roc_auc_boxplot(
    roc_auc_df: pd.DataFrame,
    model_summary: pd.DataFrame,
    models: list[ModelSpec],
) -> None:
    configured_order = [model.label for model in models]
    model_order = [model for model in configured_order if model in set(roc_auc_df["plot_model"])]
    palette = ordered_blue_model_palette(model_order)
    x_positions = [idx * 0.54 for idx in range(len(model_order))]
    grouped_values = [
        roc_auc_df.loc[roc_auc_df["plot_model"] == model, "roc_auc"].astype(float).to_numpy()
        for model in model_order
    ]
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(3.0, PLOT_HEIGHT))
    boxplot = ax.boxplot(
        grouped_values,
        positions=x_positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "0.15", "linewidth": 0.8},
        whiskerprops={"color": "0.25", "linewidth": 0.7},
        capprops={"color": "0.25", "linewidth": 0.7},
        boxprops={"edgecolor": "0.25", "linewidth": 0.7},
    )
    for patch, model in zip(boxplot["boxes"], model_order):
        patch.set_facecolor(palette[model])

    for x_position, values in zip(x_positions, grouped_values):
        jitter = rng.uniform(-0.075, 0.075, size=len(values))
        ax.scatter(
            x_position + jitter,
            values,
            color="0.25",
            alpha=0.45,
            s=5,
            linewidths=0,
        )

    ax.set_xlabel("")
    ax.set_ylabel("ROC-AUC per cell type", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        publication_model_labels(model_order),
        rotation=45,
        ha="right",
        fontsize=TICK_LABEL_SIZE,
    )
    ax.tick_params(axis="x", which="major", bottom=True, length=3, width=0.7, color="0.2")
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    if x_positions:
        ax.set_xlim(min(x_positions) - 0.32, max(x_positions) + 0.32)
    sns.despine(ax=ax)
    save_figure(fig, OUTPUT_DIR, "roc_auc_per_celltype_boxplot")


def plot_model_level_correlation(
    model_summary: pd.DataFrame,
    models: list[ModelSpec],
    metric: str,
    ylabel: str,
) -> None:
    model_order = [
        model.label
        for model in models
        if model.label in set(model_summary["plot_model"].astype(str))
    ]
    palette = ordered_blue_model_palette(model_order)
    pearson_r = model_summary["mean_roc_auc"].corr(model_summary[metric], method="pearson")
    spearman_r = model_summary["mean_roc_auc"].corr(model_summary[metric], method="spearman")

    fig, ax = plt.subplots(figsize=(2.55, PLOT_HEIGHT))
    sns.scatterplot(
        data=model_summary,
        x="mean_roc_auc",
        y=metric,
        hue="plot_model",
        hue_order=model_order,
        palette=palette,
        s=46,
        edgecolor="white",
        linewidth=0.35,
        ax=ax,
    )
    for _, row in model_summary.iterrows():
        label = str(row["plot_model"])
        ax.annotate(
            publication_model_label(label),
            (row["mean_roc_auc"], row[metric]),
            xytext=LABEL_OFFSETS.get(label, (4, 4)),
            textcoords="offset points",
            fontsize=TICK_LABEL_SIZE,
        )
    ax.set_xlabel("Mean ROC-AUC across cell types", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title(f"Pearson r={pearson_r:.2f}; Spearman r={spearman_r:.2f}", fontsize=9)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    sns.despine(ax=ax)
    save_figure(fig, OUTPUT_DIR, f"mean_roc_auc_vs_{metric}", formats=("pdf", "png", "svg"))


def main(model_names: list[str] | None = None) -> None:
    set_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models = _select_models(model_names)
    roc_auc_df, model_summary = load_ablation_data(models)
    roc_auc_df.to_csv(OUTPUT_DIR / "ablation_roc_auc_per_celltype.csv", index=False)
    model_summary.to_csv(OUTPUT_DIR / "ablation_model_level_roc_auc_metrics.csv", index=False)

    plot_roc_auc_boxplot(roc_auc_df, model_summary, models)
    for metric, ylabel in MODEL_LEVEL_METRICS:
        plot_model_level_correlation(model_summary, models, metric, ylabel)

    print(f"Saved ablation annotation plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
