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
    PUBLICATION_ABLATION_MODEL_ORDER,
    ModelSpec,
    PLOTS_DIR,
    concat_available,
    load_metrics_summary,
    ordered_blue_model_palette,
    ordered_model_labels_by_metric,
    publication_model_labels,
    save_figure,
    set_plot_style,
)


MODELS = [
    ModelSpec("CellTypist", "CellTypist_HIAI_Tcells_S2_train_AIFI_L2"),
    ModelSpec("SingleR", "SingleR_HIAI_Tcells_S2_train_AIFI_L2"),
    ModelSpec("MG", "MG", ("MG_HIAI_Tcells_S2_200k_lr5e5", "MG_HIAI_Tcells_S2_200k")),
    ModelSpec("MB", "MB", ("MB_HIAI_Tcells_S2_N1_200k_lr5e5_e15", "MB_HIAI_Tcells_S2_N1_200k_lr5e5_e15_epoch_15_ncbi")),
]
METRICS = [
    ("balanced_accuracy", "Balanced Accuracy"),
    ("accuracy", "Accuracy"),
    ("macro_f1", "Macro F1"),
    ("weighted_f1", "Weighted F1"),
]
ALIAS_COLOR_ORDER = PUBLICATION_ABLATION_MODEL_ORDER
PALETTE = {
    "CellTypist": "#54278f",
    "SingleR": "#c2a5cf",
    **ordered_blue_model_palette(ALIAS_COLOR_ORDER),
}
OUTPUT_DIR = PLOTS_DIR / "annotation_benchmark"
BAR_X_SPACING = 0.46
BAR_WIDTH = 0.22


def _select_models(model_names: list[str] | None = None) -> list[ModelSpec]:
    if not model_names:
        return MODELS
    requested = set(model_names)
    selected = [model for model in MODELS if model.label in requested]
    if not selected:
        raise ValueError(f"No annotation benchmark plot models selected from: {model_names}")
    return selected


def load_benchmark_metrics(models: list[ModelSpec]) -> pd.DataFrame:
    metrics = concat_available(
        [load_metrics_summary(model) for model in models],
        description="annotation benchmark metrics",
    )
    model_order = [model.label for model in models]
    metrics["plot_model"] = pd.Categorical(metrics["plot_model"], categories=model_order, ordered=True)
    return metrics.sort_values("plot_model")


def plot_metric(
    metrics: pd.DataFrame,
    model_order: list[str],
    metric: str,
    ylabel: str,
) -> None:
    plot_df = (
        metrics[metrics["plot_model"].astype(str).isin(model_order)]
        .copy()
    )
    plot_df["plot_model"] = pd.Categorical(
        plot_df["plot_model"].astype(str),
        categories=model_order,
        ordered=True,
    )
    plot_df = plot_df.sort_values("plot_model", kind="mergesort")
    x_positions = [idx * BAR_X_SPACING for idx in range(len(model_order))]

    fig, ax = plt.subplots(figsize=(2.25, 1.75))
    ax.bar(
        x_positions,
        plot_df[metric].astype(float),
        width=BAR_WIDTH,
        color=[PALETTE[str(model)] for model in plot_df["plot_model"]],
        edgecolor="0.15",
        linewidth=0.35,
    )
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        publication_model_labels(model_order),
        rotation=30,
        ha="right",
        fontsize=8.5,
    )
    ax.tick_params(axis="x", which="major", bottom=True, length=3, width=0.7, color="0.2")
    ax.tick_params(axis="y", labelsize=8.5)
    if x_positions:
        ax.set_xlim(min(x_positions) - 0.24, max(x_positions) + 0.24)
    sns.despine(ax=ax)
    save_figure(fig, OUTPUT_DIR, f"{metric}_benchmark")


def main(model_names: list[str] | None = None) -> None:
    set_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models = _select_models(model_names)
    metrics = load_benchmark_metrics(models)
    metrics.to_csv(OUTPUT_DIR / "annotation_benchmark_metrics.csv", index=False)
    model_order = ordered_model_labels_by_metric(
        metrics,
        model_labels=[model.label for model in models],
        metric="balanced_accuracy",
        model_column="plot_model",
        ascending=False,
    )

    for metric, ylabel in METRICS:
        plot_metric(metrics, model_order, metric, ylabel)

    print(f"Saved annotation benchmark plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
