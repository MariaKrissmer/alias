from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from alias.util.plots.color_definition import (
    PUBLICATION_ABLATION_MODEL_LABELS,
    PUBLICATION_ABLATION_MODEL_ORDER as SHARED_PUBLICATION_ABLATION_MODEL_ORDER,
    PUBLICATION_ABLATION_MODEL_PALETTE,
)

PUBLICATION_ABLATION_MODEL_ORDER = list(SHARED_PUBLICATION_ABLATION_MODEL_ORDER)
PUBLICATION_ABLATION_LABEL_MAP = dict(PUBLICATION_ABLATION_MODEL_LABELS)

FONT_FAMILY = "sans-serif"
FONT_SIZE = 9
TITLE_SIZE = 9.5
AXIS_LABEL_SIZE = 9.5
TICK_LABEL_SIZE = 8.5
LEGEND_FONT_SIZE = 8
LEGEND_TITLE_SIZE = 8
SMALL_LEGEND_FONT_SIZE = 7

PLOT_HEIGHT = 1.75
BENCHMARK_PLOT_HEIGHT = 2.3
BAR_X_SPACING = 0.46
ABLATION_BAR_X_SPACING = 0.62
BENCHMARK_BAR_X_SPACING = 0.55
BAR_WIDTH = 0.22
PAIRED_BAR_WIDTH = 0.18
SWARM_X_SPACING = 0.54
XTICK_ROTATION = 45

AXIS_EDGE_COLOR = "0.2"
AXIS_LINEWIDTH = 0.7
BAR_EDGE_COLOR = "0.15"
BAR_EDGE_LINEWIDTH = 0.35


def set_publication_style() -> None:
    """Apply the shared HIAI T-cell publication figure style."""
    sns.set_theme(
        context="paper",
        style="white",
        rc={
            "axes.edgecolor": AXIS_EDGE_COLOR,
            "axes.linewidth": AXIS_LINEWIDTH,
            "axes.grid": False,
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "legend.title_fontsize": LEGEND_TITLE_SIZE,
        },
    )


def ordered_present_models(
    models: Iterable[Any],
    *,
    reference_order: Sequence[str] = PUBLICATION_ABLATION_MODEL_ORDER,
    append_unknown: bool = True,
) -> list[str]:
    """Return present model names in publication order, preserving extras afterward."""
    present = [str(model) for model in pd.Series(list(models)).dropna().astype(str).tolist()]
    present_unique = list(dict.fromkeys(present))
    present_set = set(present_unique)
    ordered = [model for model in reference_order if model in present_set]
    if append_unknown:
        ordered.extend(model for model in present_unique if model not in set(ordered))
    return ordered


def publication_model_label(
    model: str,
    *,
    label_map: Mapping[str, str] = PUBLICATION_ABLATION_LABEL_MAP,
) -> str:
    """Map internal model IDs to publication labels."""
    return label_map.get(str(model), str(model))


def publication_model_labels(
    models: Iterable[str],
    *,
    label_map: Mapping[str, str] = PUBLICATION_ABLATION_LABEL_MAP,
) -> list[str]:
    return [publication_model_label(model, label_map=label_map) for model in models]


def publication_ablation_palette(
    models: Sequence[str],
    *,
    reference_order: Sequence[str] = PUBLICATION_ABLATION_MODEL_ORDER,
) -> dict[str, tuple[float, float, float, float]]:
    """Return consistent blue tones for internal model names."""
    cmap = plt.get_cmap("Blues")
    reference_palette = {
        model: PUBLICATION_ABLATION_MODEL_PALETTE[model]
        for model in reference_order
        if model in PUBLICATION_ABLATION_MODEL_PALETTE
    }
    fallback_order = [model for model in models if model not in reference_palette]
    fallback_values = [
        0.35 + idx * (0.45 / max(1, len(fallback_order) - 1))
        for idx in range(len(fallback_order))
    ] if fallback_order else []
    fallback_palette = {
        model: cmap(value)
        for model, value in zip(fallback_order, fallback_values)
    }
    return {
        model: reference_palette.get(model, fallback_palette.get(model, cmap(0.7)))
        for model in models
    }


def set_publication_xticklabels(
    ax: plt.Axes,
    model_order: Sequence[str],
    *,
    rotation: int = XTICK_ROTATION,
    label_map: Mapping[str, str] = PUBLICATION_ABLATION_LABEL_MAP,
) -> None:
    ax.set_xticklabels(
        publication_model_labels(model_order, label_map=label_map),
        rotation=rotation,
        ha="right",
        fontsize=TICK_LABEL_SIZE,
    )


def save_publication_figure(
    fig: plt.Figure,
    output_dir: Path | str,
    stem: str,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for file_format in formats:
        save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if file_format == "png":
            save_kwargs["dpi"] = 300
        fig.savefig(output_path / f"{stem}.{file_format}", **save_kwargs)
    plt.close(fig)
