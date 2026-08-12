from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
HIAI_TCELLS_SCRIPT_DIR = PROJECT_ROOT / "scripts" / "revision1_v1" / "HIAI_Tcells"
if str(HIAI_TCELLS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(HIAI_TCELLS_SCRIPT_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from util.publication_plotting import (  # noqa: E402
    ABLATION_BAR_X_SPACING,
    AXIS_LABEL_SIZE,
    BAR_EDGE_COLOR,
    BAR_EDGE_LINEWIDTH,
    BAR_WIDTH,
    PAIRED_BAR_WIDTH,
    PLOT_HEIGHT,
    PUBLICATION_ABLATION_LABEL_MAP,
    PUBLICATION_ABLATION_MODEL_ORDER,
    TICK_LABEL_SIZE,
    XTICK_ROTATION,
    ordered_present_models,
    publication_ablation_palette,
    publication_model_label,
    publication_model_labels,
    save_publication_figure,
    set_publication_style,
)


DATASET_ID = "S2_heldout_donor_semantic_200k"
DATASET_DIR = PROJECT_ROOT / "out" / "data" / "revision1_v1" / "HIAI_Tcells" / DATASET_ID
ANNOTATION_DIR = DATASET_DIR / "celltype_annotation"
PLOTS_DIR = ANNOTATION_DIR / "plots"


@dataclass(frozen=True)
class ModelSpec:
    label: str
    artifact_name: str
    fallback_artifact_names: tuple[str, ...] = ()


def _candidate_model_dirs(model: ModelSpec) -> list[Path]:
    return [
        ANNOTATION_DIR / artifact_name
        for artifact_name in (model.artifact_name, *model.fallback_artifact_names)
    ]


def _resolve_model_file(model: ModelSpec, file_name: str) -> Path | None:
    for model_dir in _candidate_model_dirs(model):
        path = model_dir / file_name
        if path.exists():
            return path
    return None


def set_plot_style() -> None:
    set_publication_style()


def blue_model_palette(
    model_labels: list[str],
    *,
    reference_order: list[str] | None = None,
) -> dict[str, tuple[float, float, float, float]]:
    palette_order = reference_order or model_labels
    cmap = plt.get_cmap("Blues")
    values = [0.35, 0.48, 0.61, 0.74, 0.87]
    if len(palette_order) == 1:
        values = [0.7]
    elif len(palette_order) != len(values):
        values = [
            0.35 + idx * (0.52 / max(1, len(palette_order) - 1))
            for idx in range(len(palette_order))
        ]
    full_palette = {
        label: cmap(value)
        for label, value in zip(palette_order, values)
    }
    return {label: full_palette[label] for label in model_labels if label in full_palette}


def ordered_blue_model_palette(
    model_labels: list[str],
) -> dict[str, tuple[float, float, float, float]]:
    return publication_ablation_palette(model_labels)


def ordered_model_labels_by_metric(
    df: pd.DataFrame,
    *,
    model_labels: list[str],
    metric: str = "balanced_accuracy",
    model_column: str = "plot_model",
    ascending: bool = False,
) -> list[str]:
    if model_column not in df.columns:
        return []
    present_labels = [
        label
        for label in model_labels
        if label in set(df[model_column].dropna().astype(str))
    ]
    if df.empty or metric not in df.columns or model_column not in df.columns:
        return present_labels

    model_rank = {label: idx for idx, label in enumerate(present_labels)}
    metric_df = (
        df[[model_column, metric]]
        .dropna()
        .assign(**{model_column: lambda frame: frame[model_column].astype(str)})
        .loc[lambda frame: frame[model_column].isin(present_labels)]
        .groupby(model_column, as_index=False, sort=False)[metric]
        .mean()
    )
    metric_df["_configured_order"] = metric_df[model_column].map(model_rank)
    metric_df["_configured_order"] = metric_df["_configured_order"].fillna(len(model_rank))
    metric_df = metric_df.sort_values(
        [metric, "_configured_order"],
        ascending=[ascending, True],
        kind="mergesort",
    )
    ordered = metric_df[model_column].astype(str).tolist()
    present = set(ordered)
    ordered.extend(label for label in present_labels if label not in present)
    return ordered


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    *,
    formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    save_publication_figure(fig, output_dir, stem, formats=formats)


def require_columns(df: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = columns.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def load_metrics_summary(model: ModelSpec) -> pd.DataFrame | None:
    path = _resolve_model_file(model, "metrics_summary.csv")
    if path is None:
        candidates = [str(model_dir / "metrics_summary.csv") for model_dir in _candidate_model_dirs(model)]
        print(f"Skipping {model.label}: missing any of {candidates}")
        return None
    df = pd.read_csv(path)
    require_columns(
        df,
        {"accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"},
        path,
    )
    df = df.copy()
    df["plot_model"] = model.label
    df["artifact_name"] = model.artifact_name
    return df


def load_per_celltype_recall(model: ModelSpec) -> pd.DataFrame | None:
    path = _resolve_model_file(model, "per_celltype_metrics.csv")
    if path is None:
        candidates = [str(model_dir / "per_celltype_metrics.csv") for model_dir in _candidate_model_dirs(model)]
        print(f"Skipping {model.label}: missing any of {candidates}")
        return None
    df = pd.read_csv(path)
    require_columns(df, {"cell_type", "recall"}, path)
    df = df.copy()
    df["celltype_recall"] = df["recall"]
    df["plot_model"] = model.label
    df["artifact_name"] = model.artifact_name
    return df[["plot_model", "artifact_name", "cell_type", "celltype_recall"]]


def load_label_similarity(model: ModelSpec) -> pd.DataFrame | None:
    artifact_names = (model.artifact_name, *model.fallback_artifact_names)
    candidate_paths = []
    for artifact_name in artifact_names:
        candidate_paths.extend(
            [
                ANNOTATION_DIR / artifact_name / "celltype_label_similarity.csv",
                DATASET_DIR
                / "evaluation_plots"
                / artifact_name
                / f"{artifact_name}_celltype_label_similarity.csv",
            ]
        )
    path = next((candidate for candidate in candidate_paths if candidate.exists()), None)
    if path is None:
        print(f"Skipping {model.label}: no celltype_label_similarity.csv found in configured artifacts.")
        return None

    df = pd.read_csv(path)
    require_columns(df, {"cell_type", "roc_auc"}, path)
    df = df.copy()
    df["plot_model"] = model.label
    df["artifact_name"] = model.artifact_name
    return df[["plot_model", "artifact_name", "cell_type", "roc_auc"]]


def concat_available(frames: list[pd.DataFrame | None], *, description: str) -> pd.DataFrame:
    available = [frame for frame in frames if frame is not None and not frame.empty]
    if not available:
        raise FileNotFoundError(f"No data available for {description}.")
    return pd.concat(available, ignore_index=True)
