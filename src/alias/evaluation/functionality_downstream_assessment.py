from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
import seaborn as sns

from alias.evaluation.embedding import load_dataset_embedding_artifacts, load_embedding_model
from alias.util.artifacts import write_metadata
from alias.util.plots.color_definition import (
    CATEGORICAL_PALETTES,
    COLORMAPS,
    DEG_COMPARISON_COLORS,
    DISTRIBUTION_DIFFERENCE_COLORS,
    PUBLICATION_ABLATION_MODEL_LABELS,
    PUBLICATION_ABLATION_MODEL_ORDER,
    PUBLICATION_ABLATION_MODEL_PALETTE,
    TRUE_FALSE_COLORS,
)


CutoffMethod = Literal["youden_j", "manual", "otsu", "gmm"]
SimilarityMetric = Literal["cosine", "dot"]
AssessmentScope = Literal["per_cell_type", "all_cells"]
DEFAULT_RANK_TOP_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30)
DEFAULT_RANK_TARGET_GENE_SET_FRACTION = 0.10
CYTOTOXIC_GENE_SET = ["NKG7", "GZMA", "CTSW", "CST7", "CCL5"]
REGULATORY_GENE_SET = ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TNFRSF18", "TIGIT"]
INHIBITORY_GENE_SET = ["PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "BTLA", "CD160"]
HELPER_GENE_SET = ["CD4", "IL7R", "CD40LG", "ICOS", "IL2", "IL21", "CXCR5"]
INFLAMMATORY_GENE_SET = ["IFNG", "TNF", "IL32", "CCL3", "CCL4", "NFKBIA", "IRF1"]
IMMUNOSUPPRESSIVE_GENE_SET = ["FOXP3", "IL2RA", "CTLA4", "TIGIT", "IKZF2", "TNFRSF18", "ENTPD1"]
EXHAUSTED_GENE_SET = ["PDCD1", "LAG3", "HAVCR2", "TIGIT", "TOX", "ENTPD1", "CTLA4"]
ACTIVATED_GENE_SET = ["CD69", "FOS", "JUN", "IL2RA", "TNFRSF9", "ICOS", "HLA-DRA"]
TYPE_I_INTERFERON_RESPONSE_GENE_SET = ["ISG15", "IFIT1", "IFIT2", "IFIT3", "MX1", "OAS1", "OAS2", "STAT1"]
CELL_PROLIFERATION_GENE_SET = ["MKI67", "TOP2A", "STMN1", "HMGB2", "TUBA1B", "TYMS", "PCNA"]
CYTOKINE_PRODUCTION_GENE_SET = ["IFNG", "TNF", "CCL3", "CCL4", "XCL1", "XCL2", "IL2"]
TISSUE_MIGRATION_GENE_SET = ["CCR7", "SELL", "CXCR3", "CXCR4", "CCR5", "S1PR1", "ITGA4"]
ANTIGEN_PRESENTATION_GENE_SET = ["HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "CD74", "CIITA"]
CELLULAR_STRESS_RESPONSE_GENE_SET = ["FOS", "JUN", "JUNB", "DUSP1", "HSPA1A", "HSPA1B", "DNAJB1", "ATF3"]
APOPTOTIC_SIGNALING_GENE_SET = ["BAX", "BBC3", "BCL2L11", "CASP3", "FAS", "TNFRSF10B"]

ANNOTATION_ABLATION_MODEL_ORDER = PUBLICATION_ABLATION_MODEL_ORDER
ANNOTATION_ABLATION_PUBLICATION_LABELS = PUBLICATION_ABLATION_MODEL_LABELS
ABLATION_BAR_X_SPACING = 0.62
ABLATION_BAR_WIDTH = 0.22
ABLATION_PLOT_HEIGHT = 1.75
AXIS_LABEL_SIZE = 9.5
TICK_LABEL_SIZE = 8.5
ASSOCIATED_COLORS = {
    True: TRUE_FALSE_COLORS[True],
    False: TRUE_FALSE_COLORS[False],
    "True": TRUE_FALSE_COLORS[True],
    "False": TRUE_FALSE_COLORS[False],
    "associated": TRUE_FALSE_COLORS[True],
    "unassociated": TRUE_FALSE_COLORS[False],
}
DEG_COMPARISON_BAR_WIDTH = 0.18
DEG_COMPARISON_GENE_X_SPACING = 0.46


def _publication_model_labels(model_labels: list[str]) -> list[str]:
    return [
        ANNOTATION_ABLATION_PUBLICATION_LABELS.get(str(label), str(label))
        for label in model_labels
    ]


def _set_annotation_plot_style() -> None:
    sns.set_theme(
        context="paper",
        style="white",
        rc={
            "axes.edgecolor": "0.2",
            "axes.linewidth": 0.7,
            "axes.grid": False,
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.title_fontsize": 8,
        },
    )


@dataclass(frozen=True)
class FunctionalityDownstreamConfig:
    functionality_labels: list[str]
    output_dir: Path | str
    adata_path: Path | str | None = None
    annotation_column: str = "AIFI_L2"
    cell_type_filter: list[str] | None = None
    ground_truth_column: str | None = None
    positive_values: list[Any] | None = None
    cutoff_method: CutoffMethod = "otsu"
    manual_cutoffs: dict[str, float] | None = None
    gene_sets: dict[str, list[str]] = field(default_factory=dict)
    min_group_cells: int = 30
    similarity_metric: SimilarityMetric = "cosine"
    assessment_scope: AssessmentScope = "per_cell_type"
    categorical_palette_name: str = "tab20_new"
    bins: int = 60
    cutoff_sensitivity: bool = True
    cutoff_sensitivity_window_fraction: float = 0.10
    cutoff_sensitivity_steps: int = 21
    cutoff_sensitivity_threshold_min: float | None = None
    cutoff_sensitivity_threshold_max: float | None = None
    rank_diagnostics: bool = True
    rank_top_fractions: tuple[float, ...] = DEFAULT_RANK_TOP_FRACTIONS
    rank_target_gene_set_fraction: float = DEFAULT_RANK_TARGET_GENE_SET_FRACTION
    allow_live_functionality_encoding: bool = False
    supervised_downstream_plots: bool = True
    subject_level_deg: bool = False
    subject_column: str = "subject.subjectGuid"
    pseudobulk_layer: str = "counts"
    min_subjects_per_group: int = 2
    plot: bool = True
    model_name: str | None = None
    model_name_map: dict[str, str | Path] | None = None
    timestamp: str | None = None


@dataclass(frozen=True)
class CutoffResult:
    cutoff: float
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _timestamp_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")


def _scanpy():
    import scanpy as sc

    return sc


def _safe_component(value: str) -> str:
    import re

    cleaned = re.sub(r"\s+", "_", str(value).strip())
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "", cleaned)
    return cleaned or "unknown"


def _unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = 1
    while True:
        candidate = path.with_name(f"{path.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate
        suffix += 1


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    left_norm[left_norm == 0] = 1.0
    right_norm[right_norm == 0] = 1.0
    return (left / left_norm) @ (right / right_norm).T


def _similarity(left: np.ndarray, right: np.ndarray, metric: SimilarityMetric) -> np.ndarray:
    if metric == "cosine":
        return _cosine_similarity(left, right)
    if metric == "dot":
        return left @ right.T
    raise ValueError(f"Unknown similarity metric: {metric}")


def _is_positive(value: Any, positive_values: list[Any] | None) -> bool:
    if pd.isna(value):
        return False
    if positive_values is not None:
        return value in positive_values
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"positive", "true", "1", "yes"}


def _roc_curve_numpy(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = np.asarray(y_true).astype(bool)
    values = np.asarray(scores, dtype=float)
    thresholds = np.r_[np.inf, np.sort(np.unique(values))[::-1]]
    positives = max(int(truth.sum()), 1)
    negatives = max(int((~truth).sum()), 1)
    tpr = []
    fpr = []
    for threshold in thresholds:
        predicted = values >= threshold
        tpr.append(float((predicted & truth).sum()) / positives)
        fpr.append(float((predicted & ~truth).sum()) / negatives)
    return np.asarray(fpr), np.asarray(tpr), thresholds


def _youden_j_cutoff(scores: np.ndarray, ground_truth: np.ndarray | None) -> CutoffResult:
    if ground_truth is None:
        raise ValueError("youden_j cutoff requires ground_truth labels.")
    truth = np.asarray(ground_truth).astype(bool)
    if len(np.unique(truth)) < 2:
        raise ValueError("youden_j cutoff requires both positive and negative ground_truth labels.")
    fpr, tpr, thresholds = _roc_curve_numpy(truth.astype(int), scores)
    best_idx = int(np.argmax(tpr - fpr))
    return CutoffResult(
        cutoff=float(thresholds[best_idx]),
        method="youden_j",
        metadata={"youden_j": float((tpr - fpr)[best_idx]), "deterministic": True},
    )


def _manual_cutoff(functionality_label: str, manual_cutoffs: dict[str, float] | None) -> CutoffResult:
    if not manual_cutoffs or functionality_label not in manual_cutoffs:
        raise ValueError(f"manual cutoff requires manual_cutoffs[{functionality_label!r}].")
    return CutoffResult(
        cutoff=float(manual_cutoffs[functionality_label]),
        method="manual",
        metadata={"deterministic": True},
    )


def _otsu_cutoff(scores: np.ndarray, *, n_bins: int = 256) -> CutoffResult:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("otsu cutoff requires at least two finite scores.")
    counts, edges = np.histogram(values, bins=min(n_bins, max(2, values.size)))
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    mean1 = np.cumsum(counts * centers) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((counts * centers)[::-1]) / np.maximum(weight2[::-1], 1))[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = int(np.argmax(variance12))
    return CutoffResult(
        cutoff=float(edges[1:-1][idx]),
        method="otsu",
        metadata={"n_bins": int(len(counts)), "deterministic": True},
    )


def _gmm_cutoff(scores: np.ndarray) -> CutoffResult:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)].reshape(-1, 1)
    if values.shape[0] < 2:
        raise ValueError("gmm cutoff requires at least two finite scores.")
    try:
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(n_components=2, random_state=42, n_init=10)
        model.fit(values)
        means = model.means_.ravel()
        order = np.argsort(means)
        low, high = order[0], order[1]
        mean_low, mean_high = means[low], means[high]
        std_low = float(np.sqrt(model.covariances_.reshape(-1)[low]))
        std_high = float(np.sqrt(model.covariances_.reshape(-1)[high]))
        weight_low, weight_high = model.weights_[low], model.weights_[high]

        grid = np.linspace(float(values.min()), float(values.max()), 512)
        low_pdf = weight_low * np.exp(-0.5 * ((grid - mean_low) / std_low) ** 2) / max(std_low, 1e-12)
        high_pdf = weight_high * np.exp(-0.5 * ((grid - mean_high) / std_high) ** 2) / max(std_high, 1e-12)
        between = (grid >= mean_low) & (grid <= mean_high)
        if between.any():
            idx = int(np.argmin(np.abs(low_pdf[between] - high_pdf[between])))
            cutoff = float(grid[between][idx])
        else:
            cutoff = float((mean_low + mean_high) / 2.0)
    except Exception:
        flat = np.sort(values.ravel())
        gaps = np.diff(flat)
        split = int(np.argmax(gaps))
        mean_low = float(flat[: split + 1].mean())
        mean_high = float(flat[split + 1 :].mean())
        cutoff = float((flat[split] + flat[split + 1]) / 2.0)
    return CutoffResult(
        cutoff=cutoff,
        method="gmm",
        metadata={
            "means": [float(mean_low), float(mean_high)],
            "deterministic": True,
            "random_state": 42,
            "n_init": 10,
        },
    )


def compute_cutoff(
    scores: np.ndarray | pd.Series,
    *,
    method: CutoffMethod,
    functionality_label: str,
    ground_truth: np.ndarray | pd.Series | None = None,
    manual_cutoffs: dict[str, float] | None = None,
) -> CutoffResult:
    values = np.asarray(scores, dtype=float)
    if method == "youden_j":
        return _youden_j_cutoff(values, None if ground_truth is None else np.asarray(ground_truth))
    if method == "manual":
        return _manual_cutoff(functionality_label, manual_cutoffs)
    if method == "otsu":
        return _otsu_cutoff(values)
    if method == "gmm":
        return _gmm_cutoff(values)
    raise ValueError(f"Unknown cutoff method: {method}")


def _load_adata(eval_data: Any | None, config: FunctionalityDownstreamConfig):
    if isinstance(eval_data, dict) and eval_data.get("adata") is not None:
        return eval_data["adata"].copy()
    if config.adata_path is not None:
        return _scanpy().read_h5ad(config.adata_path)
    return None


def _load_artifacts(dataset_meta: dict[str, Any], annotation_column: str) -> dict[str, Any]:
    if any(isinstance(value, dict) and "dataframe" in value for value in dataset_meta.values()):
        return {"artifacts": dataset_meta, "run_timestamp": None}
    return load_dataset_embedding_artifacts(dataset_meta, annotation_column=annotation_column)


def _embedding_columns(df: pd.DataFrame, metadata_columns: set[str]) -> list[str]:
    columns = [column for column in df.columns if column not in metadata_columns]
    numeric = df[columns].select_dtypes(include=[np.number]).columns.tolist()
    if not numeric:
        raise ValueError("No numeric embedding columns found.")
    return numeric


def _functionality_embeddings(
    loaded_artifacts: dict[str, Any],
    labels: list[str],
    *,
    model_name: str | None,
    allow_live_encoding: bool,
) -> pd.DataFrame:
    additional = loaded_artifacts.get("df_additional")
    missing = labels
    if additional is not None:
        df = additional["dataframe"].copy()
        label_column = "data" if "data" in df.columns else df.columns[0]
        available = df.set_index(label_column)
        missing = [label for label in labels if label not in available.index]
        if not missing:
            return available.loc[labels].reset_index().rename(columns={label_column: "functionality"})
    if not allow_live_encoding:
        raise ValueError(
            "Missing exact functionality embeddings for "
            f"{missing}. Recompute embeddings with these exact labels in additional_data, "
            "or set allow_live_functionality_encoding=True for exploratory runs."
        )
    if model_name is None:
        raise ValueError("Missing functionality embeddings and no model_name was provided for encoding.")
    model = load_embedding_model(model_name)
    encoded = np.asarray(model.encode(labels), dtype=np.float32)
    return pd.DataFrame(encoded).assign(functionality=labels)


def _resolve_encoding_model_name(config: FunctionalityDownstreamConfig, model_name: str) -> str | Path | None:
    if config.model_name_map and model_name in config.model_name_map:
        return config.model_name_map[model_name]
    return config.model_name or model_name


def _ordered_blue_model_palette(
    model_labels: list[str],
    *,
    reference_order: list[str] | None = None,
) -> dict[str, tuple[float, float, float, float]]:
    if all(label in PUBLICATION_ABLATION_MODEL_PALETTE for label in model_labels):
        return {label: PUBLICATION_ABLATION_MODEL_PALETTE[label] for label in model_labels}
    palette_order = reference_order or model_labels
    cmap = COLORMAPS["Blues"]
    if len(palette_order) == 1:
        values = [0.82]
    else:
        values = [
            0.92 - idx * (0.58 / max(1, len(palette_order) - 1))
            for idx in range(len(palette_order))
        ]
    full_palette = {
        label: cmap(value)
        for label, value in zip(palette_order, values)
    }
    return {label: full_palette[label] for label in model_labels if label in full_palette}


def _plot_histogram(df_sim: pd.DataFrame, cutoff: float, output_path: Path, bins: int) -> None:
    fig, ax = plt.subplots(figsize=(2.8, 1.9))
    sns.histplot(df_sim["score"], bins=_histogram_edges(df_sim["score"], bins), ax=ax)
    ax.axvline(cutoff, color="0.15", linewidth=0.9, label=f"cutoff = {cutoff:.3g}")
    ax.set_xlabel("Functionality similarity score", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Count", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.legend(frameon=False, fontsize=TICK_LABEL_SIZE, loc="best")
    sns.despine(ax=ax)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _histogram_edges(scores: pd.Series | np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(scores, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.linspace(0.0, 1.0, max(2, bins + 1))

    score_min = float(values.min())
    score_max = float(values.max())
    center = float((score_min + score_max) / 2.0)
    width = score_max - score_min
    min_width = max(np.finfo(float).eps * max(1.0, abs(center)) * bins, 1e-6)
    if not np.isfinite(width) or width < min_width:
        half_width = min_width / 2.0
        score_min = center - half_width
        score_max = center + half_width
    return np.linspace(score_min, score_max, max(2, bins + 1))


def _cutoff_sensitivity_table(
    df_sim: pd.DataFrame,
    *,
    cutoff: float,
    window_fraction: float,
    steps: int,
    threshold_min: float | None = None,
    threshold_max: float | None = None,
) -> pd.DataFrame:
    scores = np.asarray(df_sim["score"], dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return pd.DataFrame()

    score_min = float(scores.min())
    score_max = float(scores.max())
    score_range = score_max - score_min
    if not np.isfinite(score_range) or score_range <= 0:
        thresholds = np.array([float(cutoff)])
    else:
        half_window = max(0.0, float(window_fraction)) * score_range
        default_low = max(score_min, float(cutoff) - half_window)
        default_high = min(score_max, float(cutoff) + half_window)
        anchors = [float(cutoff), default_low, default_high]
        if threshold_min is not None:
            anchors.append(float(threshold_min))
        if threshold_max is not None:
            anchors.append(float(threshold_max))
        low = max(score_min, min(anchors))
        high = min(score_max, max(anchors))
        n_steps = max(3, int(steps))
        thresholds = np.linspace(low, high, n_steps)
        thresholds = np.unique(
            np.r_[
                thresholds,
                float(cutoff),
                [] if threshold_min is None else [float(threshold_min)],
                [] if threshold_max is None else [float(threshold_max)],
            ]
        )

    base_associated = np.asarray(df_sim["score"] >= cutoff, dtype=bool)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        associated = np.asarray(df_sim["score"] >= threshold, dtype=bool)
        row = {
            "threshold": float(threshold),
            "threshold_delta": float(threshold - cutoff),
            "n_associated": int(associated.sum()),
            "n_unassociated": int((~associated).sum()),
            "fraction_associated": float(associated.mean()) if associated.size else np.nan,
            "assignment_agreement_with_selected_cutoff": float((associated == base_associated).mean())
            if associated.size
            else np.nan,
        }
        if "gene_set_score" in df_sim.columns:
            valid = pd.DataFrame(
                {
                    "associated": associated,
                    "gene_set_score": df_sim["gene_set_score"].to_numpy(),
                    "score": df_sim["score"].to_numpy(),
                }
            ).dropna(subset=["gene_set_score", "score"])
            group_means = valid.groupby("associated")["gene_set_score"].mean()
            row["mean_gene_set_score_associated"] = float(group_means.get(True, np.nan))
            row["mean_gene_set_score_unassociated"] = float(group_means.get(False, np.nan))
            row["mean_gene_set_score_difference"] = float(
                group_means.get(True, np.nan) - group_means.get(False, np.nan)
            )
            row["spearman_gene_set"] = (
                float(valid["score"].corr(valid["gene_set_score"], method="spearman"))
                if len(valid) > 1
                else np.nan
            )
            row["pearson_gene_set"] = (
                float(valid["score"].corr(valid["gene_set_score"], method="pearson"))
                if len(valid) > 1
                else np.nan
            )
        rows.append(row)

    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def _percentile_ranks(values: pd.Series | np.ndarray) -> np.ndarray:
    series = pd.Series(values, dtype=float)
    if len(series) == 0:
        return np.asarray([], dtype=float)
    if len(series) == 1:
        return np.asarray([0.5], dtype=float)
    ranks = series.rank(method="average", ascending=True).to_numpy(dtype=float) - 1.0
    return ranks / float(len(series) - 1)


def _ranked_gene_set_frame(df_sim: pd.DataFrame) -> pd.DataFrame:
    if not {"score", "gene_set_score"}.issubset(df_sim.columns):
        return pd.DataFrame()
    ranked = df_sim.copy()
    ranked["score"] = pd.to_numeric(ranked["score"], errors="coerce")
    ranked["gene_set_score"] = pd.to_numeric(ranked["gene_set_score"], errors="coerce")
    ranked = ranked.dropna(subset=["score", "gene_set_score"]).copy()
    if ranked.empty:
        return ranked
    ranked["embedding_percentile_rank"] = _percentile_ranks(ranked["score"])
    ranked["gene_set_percentile_rank"] = _percentile_ranks(ranked["gene_set_score"])
    return ranked


def _area_under_metric_curve(summary: pd.DataFrame, metric_column: str) -> float:
    if summary.empty or metric_column not in summary:
        return np.nan
    ordered = summary.dropna(subset=["top_fraction", metric_column]).sort_values("top_fraction")
    if ordered.empty:
        return np.nan
    if len(ordered) == 1:
        return float(ordered[metric_column].iloc[0])
    x = ordered["top_fraction"].to_numpy(dtype=float)
    y = ordered[metric_column].to_numpy(dtype=float)
    width = float(x[-1] - x[0])
    if width <= 0:
        return float(np.nanmean(y))
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(y, x) / width)


def _rank_tail_enrichment_table(
    ranked: pd.DataFrame,
    *,
    top_fractions: tuple[float, ...],
) -> pd.DataFrame:
    if ranked.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    n_cells = int(len(ranked))
    for top_fraction in top_fractions:
        fraction = float(top_fraction)
        if fraction <= 0 or fraction > 1:
            raise ValueError("rank_top_fractions must be greater than 0 and at most 1.")
        n_top = max(1, round(n_cells * fraction))
        embedding_top = ranked.nlargest(n_top, "score")
        gene_set_top_ids = set(ranked.nlargest(n_top, "gene_set_score").index)
        n_overlap = int(sum(index in gene_set_top_ids for index in embedding_top.index))
        overlap_fraction = float(n_overlap / n_top) if n_top else np.nan
        random_expectation = float(n_top / n_cells) if n_cells else np.nan
        lift = float(overlap_fraction / random_expectation) if random_expectation else np.nan
        rows.append(
            {
                "top_fraction": fraction,
                "n_cells": n_cells,
                "n_embedding_top_cells": int(n_top),
                "n_gene_set_top_cells": int(n_top),
                "n_overlap": n_overlap,
                "overlap_fraction": overlap_fraction,
                "random_expectation": random_expectation,
                "lift": lift,
                "median_gene_set_percentile_rank": float(
                    embedding_top["gene_set_percentile_rank"].median()
                ),
                "spearman_in_embedding_top": (
                    float(embedding_top["score"].corr(embedding_top["gene_set_score"], method="spearman"))
                    if len(embedding_top) > 1
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _rank_tail_summary(tail_table: pd.DataFrame) -> dict[str, Any]:
    if tail_table.empty:
        return {
            "area_under_lift_curve": np.nan,
            "area_under_median_gene_set_percentile_curve": np.nan,
            "mean_median_gene_set_percentile_rank": np.nan,
            "mean_spearman_in_embedding_top": np.nan,
            "lift_at_10_percent": np.nan,
            "overlap_fraction_at_10_percent": np.nan,
        }
    ordered = tail_table.sort_values("top_fraction").reset_index(drop=True)
    closest_10 = ordered.loc[ordered["top_fraction"].sub(0.10).abs().idxmin()]
    return {
        "area_under_lift_curve": _area_under_metric_curve(ordered, "lift"),
        "area_under_median_gene_set_percentile_curve": _area_under_metric_curve(
            ordered,
            "median_gene_set_percentile_rank",
        ),
        "mean_median_gene_set_percentile_rank": float(ordered["median_gene_set_percentile_rank"].mean()),
        "mean_spearman_in_embedding_top": float(ordered["spearman_in_embedding_top"].mean()),
        "lift_at_10_percent": float(closest_10["lift"]),
        "overlap_fraction_at_10_percent": float(closest_10["overlap_fraction"]),
    }


def _rank_bins(values: pd.Series, bins: int) -> pd.Series:
    ranked_positions = values.astype(float).sort_values(kind="mergesort").index.to_list()
    result = pd.Series(index=values.index, dtype=int)
    n_rows = len(ranked_positions)
    for bin_index in range(bins):
        start = round(bin_index * n_rows / bins)
        stop = round((bin_index + 1) * n_rows / bins)
        for row_index in ranked_positions[start:stop]:
            result.loc[row_index] = bin_index + 1
    return result.astype(int)


def _rank_decile_overlap_table(ranked: pd.DataFrame, *, bins: int = 10) -> pd.DataFrame:
    if ranked.empty:
        return pd.DataFrame()
    work = ranked[["score", "gene_set_score"]].copy()
    work["embedding_decile"] = _rank_bins(work["score"], bins)
    work["gene_set_decile"] = _rank_bins(work["gene_set_score"], bins)
    embedding_totals = work["embedding_decile"].value_counts().to_dict()
    gene_set_totals = work["gene_set_decile"].value_counts().to_dict()
    counts = work.groupby(["embedding_decile", "gene_set_decile"]).size().to_dict()
    rows: list[dict[str, Any]] = []
    n_rows = len(work)
    for embedding_decile in range(1, bins + 1):
        for gene_set_decile in range(1, bins + 1):
            n_cells = int(counts.get((embedding_decile, gene_set_decile), 0))
            rows.append(
                {
                    "embedding_decile": embedding_decile,
                    "gene_set_decile": gene_set_decile,
                    "n_cells": n_cells,
                    "fraction_of_all_cells": float(n_cells / n_rows) if n_rows else np.nan,
                    "fraction_of_embedding_decile": float(n_cells / embedding_totals[embedding_decile])
                    if embedding_totals.get(embedding_decile)
                    else np.nan,
                    "fraction_of_gene_set_decile": float(n_cells / gene_set_totals[gene_set_decile])
                    if gene_set_totals.get(gene_set_decile)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _rank_cell_type_tail_summary(
    ranked: pd.DataFrame,
    *,
    annotation_column: str,
    top_fraction: float,
    max_types: int = 10,
) -> pd.DataFrame:
    if ranked.empty or annotation_column not in ranked:
        return pd.DataFrame()
    n_top = max(1, round(len(ranked) * float(top_fraction)))
    top = ranked.nlargest(n_top, "score")
    if top.empty:
        return pd.DataFrame()
    summary = (
        top.groupby(annotation_column, dropna=False)
        .agg(
            n_cells=(annotation_column, "size"),
            median_gene_set_percentile_rank=("gene_set_percentile_rank", "median"),
            median_embedding_percentile_rank=("embedding_percentile_rank", "median"),
        )
        .reset_index()
        .sort_values(["n_cells", "median_gene_set_percentile_rank"], ascending=[False, False])
        .head(max_types)
    )
    summary["fraction_of_embedding_tail"] = summary["n_cells"] / len(top)
    return summary


def _rank_diagnostic_outputs(
    df_sim: pd.DataFrame,
    *,
    annotation_column: str,
    top_fractions: tuple[float, ...],
    target_gene_set_fraction: float,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ranked = _ranked_gene_set_frame(df_sim)
    tail_table = _rank_tail_enrichment_table(ranked, top_fractions=top_fractions)
    summary = _rank_tail_summary(tail_table)
    decile_overlap = _rank_decile_overlap_table(ranked, bins=10)
    cell_type_tail = _rank_cell_type_tail_summary(
        ranked,
        annotation_column=annotation_column,
        top_fraction=target_gene_set_fraction,
    )
    return tail_table, summary, decile_overlap, cell_type_tail, ranked


def _plot_cutoff_sensitivity(sensitivity: pd.DataFrame, output_path: Path, *, cutoff: float) -> None:
    if sensitivity.empty or "threshold" not in sensitivity:
        return

    fig, ax = plt.subplots(figsize=(2.8, 1.9))
    ax.plot(
        sensitivity["threshold"],
        sensitivity["fraction_associated"],
        color="0.2",
        linewidth=0.9,
    )
    ax.scatter(
        sensitivity["threshold"],
        sensitivity["fraction_associated"],
        color="0.2",
        s=8,
        linewidth=0,
        zorder=3,
    )
    ax.axvline(cutoff, color=TRUE_FALSE_COLORS[True], linewidth=0.9, label=f"selected cutoff = {cutoff:.3g}")
    ax.set_xlabel("Cutoff", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Associated fraction", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.legend(frameon=False, fontsize=TICK_LABEL_SIZE, loc="best")
    sns.despine(ax=ax)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _expression_frame_for_genes(adata: Any, cell_index: pd.Index, genes: list[str]) -> pd.DataFrame:
    if adata is None or not genes or not hasattr(adata, "var_names") or not hasattr(adata, "X"):
        return pd.DataFrame()

    adata_index = pd.Index(adata.obs.index.astype(str))
    shared = cell_index.astype(str).intersection(adata_index)
    present_genes = [gene for gene in genes if gene in set(map(str, adata.var_names))]
    if len(shared) == 0 or not present_genes:
        return pd.DataFrame()

    row_positions = adata_index.get_indexer(shared)
    var_index = pd.Index(adata.var_names.astype(str))
    gene_positions = var_index.get_indexer(present_genes)
    matrix = adata.X[row_positions][:, gene_positions]
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    expression = pd.DataFrame(
        np.asarray(matrix, dtype=float),
        index=shared,
        columns=present_genes,
    )
    return expression


def _top_deg_cutoff_sensitivity_table(
    adata: Any,
    df_sim: pd.DataFrame,
    deg_summary: pd.DataFrame,
    sensitivity: pd.DataFrame,
    *,
    top_n: int = 5,
) -> pd.DataFrame:
    if sensitivity.empty or deg_summary.empty or "top_genes" not in deg_summary:
        return pd.DataFrame()
    top_gene_value = deg_summary.loc[0, "top_genes"]
    if pd.isna(top_gene_value):
        return pd.DataFrame()
    top_genes = [gene for gene in str(top_gene_value).split(";") if gene][:top_n]
    expression = _expression_frame_for_genes(adata, df_sim.index, top_genes)
    if expression.empty:
        return pd.DataFrame()

    scores = df_sim.loc[expression.index, "score"].astype(float)
    rows: list[dict[str, Any]] = []
    for threshold in sensitivity["threshold"].astype(float):
        associated = scores >= threshold
        n_associated = int(associated.sum())
        n_unassociated = int((~associated).sum())
        for gene in expression.columns:
            associated_mean = float(expression.loc[associated, gene].mean()) if n_associated else np.nan
            unassociated_mean = float(expression.loc[~associated, gene].mean()) if n_unassociated else np.nan
            rows.append(
                {
                    "gene": gene,
                    "threshold": float(threshold),
                    "n_associated": n_associated,
                    "n_unassociated": n_unassociated,
                    "mean_expression_associated": associated_mean,
                    "mean_expression_unassociated": unassociated_mean,
                    "mean_expression_difference": float(associated_mean - unassociated_mean),
                }
            )
    return pd.DataFrame(rows)


def _plot_top_deg_cutoff_sensitivity(sensitivity: pd.DataFrame, output_path: Path, *, cutoff: float) -> None:
    if sensitivity.empty or not {"gene", "threshold", "mean_expression_difference"}.issubset(sensitivity.columns):
        return

    genes = list(dict.fromkeys(sensitivity["gene"].astype(str).tolist()))
    cmap = COLORMAPS["Reds"]
    if len(genes) == 1:
        color_values = [0.75]
    else:
        color_values = np.linspace(0.35, 0.90, len(genes))
    color_map = {gene: cmap(value) for gene, value in zip(genes, color_values)}
    fig, ax = plt.subplots(figsize=(3.0, 2.1))
    line_ends: list[tuple[str, float, float]] = []
    for gene in genes:
        gene_df = sensitivity[sensitivity["gene"].astype(str) == gene].sort_values("threshold")
        ax.plot(
            gene_df["threshold"],
            gene_df["mean_expression_difference"],
            color=color_map[gene],
            linewidth=0.9,
            label="_nolegend_",
        )
        ax.scatter(
            gene_df["threshold"],
            gene_df["mean_expression_difference"],
            color=color_map[gene],
            s=8,
            linewidth=0,
            zorder=3,
        )
        if not gene_df.empty:
            line_ends.append(
                (
                    gene,
                    float(gene_df["threshold"].iloc[-1]),
                    float(gene_df["mean_expression_difference"].iloc[-1]),
                )
            )
    y_values = sensitivity["mean_expression_difference"].replace([np.inf, -np.inf], np.nan).dropna()
    y_range = float(y_values.max() - y_values.min()) if not y_values.empty else 1.0
    min_gap = max(0.035 * max(y_range, 1e-6), 1e-6)
    adjusted_labels: list[tuple[str, float, float]] = []
    for gene, x_value, y_value in sorted(line_ends, key=lambda item: item[2]):
        adjusted_y = y_value
        if adjusted_labels and adjusted_y - adjusted_labels[-1][2] < min_gap:
            adjusted_y = adjusted_labels[-1][2] + min_gap
        adjusted_labels.append((gene, x_value, adjusted_y))
    x_values = sensitivity["threshold"].replace([np.inf, -np.inf], np.nan).dropna()
    x_range = float(x_values.max() - x_values.min()) if len(x_values) > 1 else 1.0
    label_x_offset = max(0.025 * max(x_range, 1e-6), 1e-6)
    for gene, x_value, y_value in adjusted_labels:
        ax.text(
            x_value + label_x_offset,
            y_value,
            gene,
            color=color_map[gene],
            fontsize=TICK_LABEL_SIZE,
            va="center",
            ha="left",
            clip_on=False,
        )
    ax.axvline(cutoff, color="0.15", linewidth=0.8, label=f"selected cutoff = {cutoff:.3g}")
    ax.axhline(0, color="0.7", linewidth=0.45)
    ax.set_xlabel("Cutoff", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Mean expression difference", fontsize=AXIS_LABEL_SIZE)
    if len(x_values):
        tick_start = np.ceil(float(x_values.min()) * 10) / 10
        tick_end = np.floor(float(x_values.max()) * 10) / 10
        if tick_start <= tick_end:
            ax.set_xticks(np.round(np.arange(tick_start, tick_end + 0.001, 0.1), 1))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=TICK_LABEL_SIZE,
    )
    if len(x_values):
        ax.set_xlim(float(x_values.min()), float(x_values.max()) + 0.22 * max(x_range, 1e-6))
    sns.despine(ax=ax)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _gene_cutoff_sensitivity_summary(
    adata: Any,
    df_sim: pd.DataFrame,
    sensitivity: pd.DataFrame,
    *,
    cutoff: float,
    gene_chunk_size: int = 512,
) -> pd.DataFrame:
    if adata is None or sensitivity.empty or not hasattr(adata, "var_names"):
        return pd.DataFrame()
    genes = list(map(str, adata.var_names))
    thresholds = sensitivity["threshold"].astype(float).dropna().sort_values().unique()
    if len(thresholds) < 2:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for start in range(0, len(genes), max(1, int(gene_chunk_size))):
        chunk_genes = genes[start : start + max(1, int(gene_chunk_size))]
        expression = _expression_frame_for_genes(adata, df_sim.index, chunk_genes)
        if expression.empty:
            continue
        scores = df_sim.loc[expression.index, "score"].astype(float)
        for gene in expression.columns:
            values = expression[gene].astype(float)
            rows: list[dict[str, float]] = []
            for threshold in thresholds:
                associated = scores >= threshold
                n_associated = int(associated.sum())
                n_unassociated = int((~associated).sum())
                associated_mean = float(values.loc[associated].mean()) if n_associated else np.nan
                unassociated_mean = float(values.loc[~associated].mean()) if n_unassociated else np.nan
                rows.append(
                    {
                        "threshold": float(threshold),
                        "n_associated": n_associated,
                        "n_unassociated": n_unassociated,
                        "mean_expression_difference": float(associated_mean - unassociated_mean),
                    }
                )
            gene_df = pd.DataFrame(rows).dropna(subset=["mean_expression_difference"])
            if len(gene_df) < 2:
                continue
            x = gene_df["threshold"].to_numpy(dtype=float)
            y = gene_df["mean_expression_difference"].to_numpy(dtype=float)
            slope = float(np.polyfit(x, y, deg=1)[0])
            low_idx = int(np.argmin(x))
            high_idx = int(np.argmax(x))
            selected_idx = int(np.argmin(np.abs(x - cutoff)))
            spearman = (
                np.nan
                if np.nanstd(y) == 0
                else float(pd.Series(x).corr(pd.Series(y), method="spearman"))
            )
            records.append(
                {
                    "gene": gene,
                    "slope": slope,
                    "delta_low_to_high": float(y[high_idx] - y[low_idx]),
                    "max_difference": float(np.nanmax(y)),
                    "min_difference": float(np.nanmin(y)),
                    "spearman_with_cutoff": spearman,
                    "n_valid_thresholds": int(len(gene_df)),
                    "difference_at_lowest_threshold": float(y[low_idx]),
                    "difference_at_highest_threshold": float(y[high_idx]),
                    "difference_at_selected_cutoff": float(y[selected_idx]),
                }
            )

    if not records:
        return pd.DataFrame()
    return (
        pd.DataFrame(records)
        .sort_values(["slope", "delta_low_to_high", "max_difference"], ascending=[False, False, False])
        .reset_index(drop=True)
    )


def _plot_gene_cutoff_sensitivity_top_increasing(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    top_n: int = 20,
) -> None:
    if summary.empty or not {"gene", "slope"}.issubset(summary.columns):
        return
    plot_df = summary.dropna(subset=["slope"]).head(top_n).copy()
    if plot_df.empty:
        return
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)
    cmap = COLORMAPS["Reds"]
    color_values = np.linspace(0.35, 0.9, len(plot_df)) if len(plot_df) > 1 else [0.75]
    fig_height = max(2.2, 0.15 * len(plot_df) + 0.8)
    fig, ax = plt.subplots(figsize=(3.1, fig_height))
    ax.barh(
        plot_df["gene"],
        plot_df["slope"],
        color=[cmap(value) for value in color_values],
        edgecolor="0.15",
        linewidth=0.25,
    )
    ax.axvline(0, color="0.45", linewidth=0.55)
    ax.set_xlabel("Slope of expression contrast", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    sns.despine(ax=ax)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _plot_histogram_by_cell_type(
    df_sim: pd.DataFrame,
    *,
    annotation_column: str,
    cutoff: float,
    output_path: Path,
    bins: int,
    palette_name: str = "tab20_new",
    show_cutoff: bool = True,
) -> None:
    if annotation_column not in df_sim:
        return
    labels = sorted(df_sim[annotation_column].dropna().astype(str).unique())
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(3.6, 2.45))
    base_palette = CATEGORICAL_PALETTES.get(palette_name, CATEGORICAL_PALETTES["tab20_new"])
    palette = [base_palette[index % len(base_palette)] for index in range(len(labels))]
    bin_edges = _histogram_edges(df_sim["score"], bins)

    for index, cell_type in enumerate(labels):
        values = df_sim.loc[df_sim[annotation_column].astype(str) == cell_type, "score"]
        ax.hist(
            values,
            bins=bin_edges,
            alpha=0.45,
            density=True,
            label=cell_type,
            color=palette[index],
            histtype="stepfilled",
        )
    if show_cutoff:
        ax.axvline(cutoff, color="0.15", linewidth=0.9, label=f"cutoff = {cutoff:.3g}")
    ax.set_xlabel("Functionality similarity score", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Density", fontsize=AXIS_LABEL_SIZE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        length=4,
        labelsize=TICK_LABEL_SIZE,
    )
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=max(1, min(4, (len(labels) + 1) // 2)),
        fontsize=TICK_LABEL_SIZE,
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter(df_sim: pd.DataFrame, output_path: Path) -> None:
    if "gene_set_score" not in df_sim.columns:
        return
    fig, ax = plt.subplots(figsize=(3, 2))
    sns.scatterplot(
        data=df_sim,
        x="score",
        y="gene_set_score",
        hue="associated",
        palette=ASSOCIATED_COLORS,
        s=12,
        ax=ax,
    )
    sns.despine(ax=ax)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_model_gene_set_scatter_grid(frames: list[pd.DataFrame], output_path: Path) -> None:
    if not frames:
        return
    plot_df = pd.concat(frames, ignore_index=True)
    plot_df = plot_df.dropna(subset=["score", "gene_set_score", "model_name"])
    if plot_df.empty:
        return

    models = list(dict.fromkeys(plot_df["model_name"].astype(str)))
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.6 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)

    for index, model_name in enumerate(models):
        ax = axes.ravel()[index]
        ax.set_visible(True)
        model_df = plot_df[plot_df["model_name"].astype(str) == model_name]
        sns.scatterplot(
            data=model_df,
            x="score",
            y="gene_set_score",
            hue="associated",
            palette=ASSOCIATED_COLORS,
            s=6,
            linewidth=0,
            alpha=0.6,
            ax=ax,
        )
        rho = model_df["score"].corr(model_df["gene_set_score"], method="spearman") if len(model_df) > 1 else np.nan
        ax.set_title(f"{model_name} rho={rho:.2f}" if pd.notna(rho) else str(model_name), fontsize=8)
        ax.set_xlabel("Similarity score")
        ax.set_ylabel("Gene-set score")
        ax.legend(frameon=False, fontsize=6, title=None, loc="best")
        sns.despine(ax=ax)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_gene_set_correlation_barplot(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    metric_column: str = "spearman_gene_set",
    y_label: str = "Spearman correlation",
) -> None:
    if summary.empty or metric_column not in summary:
        return
    plot_df = summary.dropna(subset=[metric_column]).copy()
    if plot_df.empty:
        return

    plot_df["model_name"] = plot_df["model_name"].astype(str)
    if plot_df["functionality"].astype(str).nunique() > 1:
        return
    model_order = [
        model for model in ANNOTATION_ABLATION_MODEL_ORDER if model in set(plot_df["model_name"])
    ]
    model_order.extend(
        model for model in plot_df["model_name"].tolist() if model not in set(model_order)
    )
    plot_df = (
        plot_df.sort_values("model_name", key=lambda series: series.map({model: idx for idx, model in enumerate(model_order)}))
        .drop_duplicates(subset=["model_name", "functionality"], keep="first")
        .set_index("model_name")
        .loc[model_order]
        .reset_index()
    )
    palette = _ordered_blue_model_palette(model_order, reference_order=ANNOTATION_ABLATION_MODEL_ORDER)
    x_positions = np.arange(len(model_order)) * ABLATION_BAR_X_SPACING

    fig, ax = plt.subplots(figsize=(3.25, ABLATION_PLOT_HEIGHT))
    ax.bar(
        x_positions,
        plot_df[metric_column].astype(float),
        width=ABLATION_BAR_WIDTH,
        color=[palette[model] for model in model_order],
        edgecolor="0.15",
        linewidth=0.35,
    )
    ax.set_xlabel("")
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(-1, 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        _publication_model_labels(model_order),
        rotation=45,
        ha="right",
        fontsize=TICK_LABEL_SIZE,
    )
    ax.tick_params(axis="x", which="major", bottom=True, length=3, width=0.7, color="0.2")
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    if len(x_positions):
        ax.set_xlim(min(x_positions) - 0.36, max(x_positions) + 0.36)
    sns.despine(ax=ax)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_gene_set_correlation_dotplot(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    metric_column: str = "spearman_gene_set",
    y_label: str = "Spearman correlation",
) -> None:
    if summary.empty or metric_column not in summary:
        return
    plot_df = summary.dropna(subset=[metric_column]).copy()
    if plot_df.empty:
        return

    plot_df["model_name"] = plot_df["model_name"].astype(str)
    plot_df["functionality"] = plot_df["functionality"].astype(str)
    model_order = [
        model for model in ANNOTATION_ABLATION_MODEL_ORDER if model in set(plot_df["model_name"])
    ]
    model_order.extend(
        model for model in plot_df["model_name"].tolist() if model not in set(model_order)
    )
    functionality_order = list(dict.fromkeys(summary["functionality"].astype(str).tolist()))
    functionality_order = [label for label in functionality_order if label in set(plot_df["functionality"])]
    if not model_order or not functionality_order:
        return

    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    markers = {
        label: marker_cycle[index % len(marker_cycle)]
        for index, label in enumerate(functionality_order)
    }
    palette = _ordered_blue_model_palette(model_order, reference_order=ANNOTATION_ABLATION_MODEL_ORDER)
    x_positions = np.arange(len(model_order)) * ABLATION_BAR_X_SPACING
    x_lookup = {model: position for model, position in zip(model_order, x_positions)}
    offsets = np.linspace(-0.13, 0.13, len(functionality_order)) if len(functionality_order) > 1 else np.array([0.0])
    offset_lookup = {
        label: float(offset)
        for label, offset in zip(functionality_order, offsets)
    }

    width = max(3.25, 0.48 * len(model_order))
    fig, ax = plt.subplots(figsize=(width, ABLATION_PLOT_HEIGHT))
    for functionality in functionality_order:
        function_df = plot_df[plot_df["functionality"] == functionality]
        for _, row in function_df.iterrows():
            model = row["model_name"]
            if model not in x_lookup or model not in palette:
                continue
            ax.scatter(
                x_lookup[model] + offset_lookup[functionality],
                float(row[metric_column]),
                marker=markers[functionality],
                s=28,
                facecolor=palette[model],
                edgecolor="0.15",
                linewidth=0.45,
                zorder=3,
            )

    ax.set_xlabel("")
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="0.65", linewidth=0.45, zorder=1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        _publication_model_labels(model_order),
        rotation=45,
        ha="right",
        fontsize=TICK_LABEL_SIZE,
    )
    ax.tick_params(axis="x", which="major", bottom=True, length=3, width=0.7, color="0.2")
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    if len(x_positions):
        ax.set_xlim(min(x_positions) - 0.42, max(x_positions) + 0.42)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=markers[label],
            color="none",
            markerfacecolor="0.72",
            markeredgecolor="0.15",
            markeredgewidth=0.45,
            markersize=5,
            linestyle="None",
            label=label,
        )
        for label in functionality_order
    ]
    ax.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.36),
        ncol=max(1, min(3, len(legend_handles))),
        fontsize=TICK_LABEL_SIZE,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    sns.despine(ax=ax)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _rank_plot_model_order(values: pd.Series) -> list[str]:
    present = list(dict.fromkeys(values.dropna().astype(str).tolist()))
    ordered = [model for model in ANNOTATION_ABLATION_MODEL_ORDER if model in set(present)]
    ordered.extend(model for model in present if model not in set(ordered))
    return ordered


def _plot_rank_tail_metric_line(
    metrics: pd.DataFrame,
    output_path: Path,
    *,
    metric_column: str,
    y_label: str,
    reference_line: float,
) -> None:
    if metrics.empty or metric_column not in metrics:
        return
    plot_df = metrics.dropna(subset=["top_fraction", metric_column, "model_name"]).copy()
    if plot_df.empty:
        return
    plot_df["model_name"] = plot_df["model_name"].astype(str)
    plot_df["functionality"] = plot_df["functionality"].astype(str)
    model_order = _rank_plot_model_order(plot_df["model_name"])
    functionality_order = list(dict.fromkeys(plot_df["functionality"].tolist()))
    palette = _ordered_blue_model_palette(model_order, reference_order=ANNOTATION_ABLATION_MODEL_ORDER)
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    marker_lookup = {
        functionality: marker_cycle[index % len(marker_cycle)]
        for index, functionality in enumerate(functionality_order)
    }

    width = max(2.8, 0.42 * len(model_order) + 0.9)
    fig, ax = plt.subplots(figsize=(width, 2.15))
    for functionality in functionality_order:
        for model in model_order:
            subset = plot_df[
                (plot_df["model_name"] == model)
                & (plot_df["functionality"] == functionality)
            ].sort_values("top_fraction")
            if subset.empty:
                continue
            label = _publication_model_labels([model])[0]
            if len(functionality_order) > 1:
                label = f"{label} / {functionality}"
            ax.plot(
                subset["top_fraction"],
                subset[metric_column],
                color=palette[model],
                marker=marker_lookup[functionality],
                markersize=3.2,
                linewidth=1.0,
                label=label,
            )

    ax.axhline(reference_line, color="0.55", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Top embedding-score fraction", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    ax.legend(
        frameon=False,
        fontsize=TICK_LABEL_SIZE,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    sns.despine(ax=ax)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_rank_tail_metric_bar(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    metric_column: str,
    y_label: str,
    reference_line: float,
) -> None:
    if summary.empty or metric_column not in summary:
        return
    plot_df = summary.dropna(subset=[metric_column, "model_name"]).copy()
    if plot_df.empty:
        return
    plot_df["model_name"] = plot_df["model_name"].astype(str)
    plot_df["functionality"] = plot_df["functionality"].astype(str)
    model_order = _rank_plot_model_order(plot_df["model_name"])
    functionality_order = list(dict.fromkeys(plot_df["functionality"].tolist()))
    palette = _ordered_blue_model_palette(model_order, reference_order=ANNOTATION_ABLATION_MODEL_ORDER)
    x_positions = np.arange(len(model_order)) * ABLATION_BAR_X_SPACING
    offsets = np.linspace(-0.14, 0.14, len(functionality_order)) if len(functionality_order) > 1 else np.array([0.0])
    width = min(ABLATION_BAR_WIDTH, 0.25 / max(1, len(functionality_order)))

    fig, ax = plt.subplots(figsize=(max(3.25, 0.48 * len(model_order) + 0.7), ABLATION_PLOT_HEIGHT))
    for functionality, offset in zip(functionality_order, offsets):
        values = []
        positions = []
        colors = []
        for model, x_pos in zip(model_order, x_positions):
            rows = plot_df[
                (plot_df["model_name"] == model)
                & (plot_df["functionality"] == functionality)
            ]
            if rows.empty:
                continue
            values.append(float(rows[metric_column].iloc[0]))
            positions.append(float(x_pos + offset))
            colors.append(palette[model])
        if not values:
            continue
        ax.bar(
            positions,
            values,
            width=width,
            color=colors,
            edgecolor="0.15",
            linewidth=0.35,
            label=functionality if len(functionality_order) > 1 else None,
        )
    ax.axhline(reference_line, color="0.55", linewidth=0.6, linestyle="--")
    ax.set_xlabel("")
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        _publication_model_labels(model_order),
        rotation=45,
        ha="right",
        fontsize=TICK_LABEL_SIZE,
    )
    ax.tick_params(axis="x", which="major", bottom=True, length=3, width=0.7, color="0.2")
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    if len(x_positions):
        ax.set_xlim(min(x_positions) - 0.36, max(x_positions) + 0.36)
    if len(functionality_order) > 1:
        ax.legend(frameon=False, fontsize=TICK_LABEL_SIZE, loc="center left", bbox_to_anchor=(1.02, 0.5))
    sns.despine(ax=ax)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_rank_decile_overlap_heatmap(overlap: pd.DataFrame, output_path: Path) -> None:
    if overlap.empty:
        return
    required_columns = {
        "embedding_decile",
        "gene_set_decile",
        "fraction_of_embedding_decile",
    }
    if not required_columns.issubset(overlap.columns):
        return

    heatmap_data = (
        overlap.pivot(
            index="gene_set_decile",
            columns="embedding_decile",
            values="fraction_of_embedding_decile",
        )
        .reindex(index=range(10, 0, -1), columns=range(1, 11))
        .astype(float)
    )

    fig, ax = plt.subplots(figsize=(3.9, 3.55))
    sns.heatmap(
        heatmap_data,
        ax=ax,
        cmap=COLORMAPS["Reds"],
        vmin=0,
        square=True,
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Fraction within embedding decile", "shrink": 0.75},
    )
    ax.set_title("Decile overlap", fontsize=AXIS_LABEL_SIZE + 2.5)
    ax.set_xlabel("Embedding-score decile", fontsize=AXIS_LABEL_SIZE + 1.5)
    ax.set_ylabel("Gene-set-score decile", fontsize=AXIS_LABEL_SIZE + 1.5)
    ax.set_xticklabels(range(1, 11), rotation=0, fontsize=TICK_LABEL_SIZE + 1.5)
    ax.set_yticklabels(range(10, 0, -1), rotation=0, fontsize=TICK_LABEL_SIZE + 1.5)
    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=TICK_LABEL_SIZE + 1.0)
        cbar.set_label("Fraction within embedding decile", fontsize=AXIS_LABEL_SIZE + 1.0)
        cbar.outline.set_visible(True)
        cbar.outline.set_edgecolor("black")
        cbar.outline.set_linewidth(0.45)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_umap(
    df_sim: pd.DataFrame,
    output_path: Path,
    color_column: str,
    *,
    continuous: bool,
    x_column: str = "UMAP1",
    y_column: str = "UMAP2",
) -> None:
    if not {x_column, y_column, color_column}.issubset(df_sim.columns):
        return
    plot_df = df_sim[[x_column, y_column, color_column]].dropna()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(3, 3))
    if continuous:
        points = ax.scatter(
            plot_df[x_column],
            plot_df[y_column],
            c=plot_df[color_column],
            cmap="viridis",
            s=3,
            linewidths=0,
            rasterized=True,
        )
        fig.colorbar(points, ax=ax, fraction=0.046, pad=0.04)
    else:
        sns.scatterplot(
            data=plot_df.assign(**{color_column: plot_df[color_column].astype(str)}),
            x=x_column,
            y=y_column,
            hue=color_column,
            palette=ASSOCIATED_COLORS
            if color_column in {"associated", "ground_truth_positive"}
            else CATEGORICAL_PALETTES.get("tab20_new"),
            s=5,
            linewidth=0,
            ax=ax,
        )
        ax.legend(frameon=False, fontsize=6, title=None, loc="best")
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    sns.despine(ax=ax)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _attach_umap_coordinates(
    work: pd.DataFrame,
    *,
    cell_artifact: dict[str, Any],
    adata: Any | None,
) -> pd.DataFrame:
    result = work.copy()
    has_default_umap = False
    umap_df = cell_artifact.get("umap")
    if isinstance(umap_df, pd.DataFrame) and not umap_df.empty:
        coords = umap_df.copy()
        if {"UMAP1", "UMAP2"}.issubset(coords.columns):
            coords = coords[["UMAP1", "UMAP2"]]
        elif coords.shape[1] >= 2:
            coords = coords.iloc[:, :2]
            coords.columns = ["UMAP1", "UMAP2"]
        else:
            coords = pd.DataFrame()
        if not coords.empty:
            shared = result.index.intersection(coords.index.astype(str))
            if len(shared) > 0:
                coords.index = coords.index.astype(str)
                values = coords.loc[shared, ["UMAP1", "UMAP2"]].to_numpy()
                result.loc[shared, ["model_UMAP1", "model_UMAP2"]] = values
                result.loc[shared, ["UMAP1", "UMAP2"]] = values
                has_default_umap = True

    if adata is not None and hasattr(adata, "obsm") and "X_umap" in adata.obsm:
        coords = np.asarray(adata.obsm["X_umap"])
        if coords.ndim == 2 and coords.shape[1] >= 2:
            adata_index = pd.Index(adata.obs.index.astype(str))
            coord_df = pd.DataFrame(coords[:, :2], index=adata_index, columns=["UMAP1", "UMAP2"])
            shared = result.index.intersection(coord_df.index)
            if len(shared) > 0:
                values = coord_df.loc[shared, ["UMAP1", "UMAP2"]].to_numpy()
                result.loc[shared, ["adata_UMAP1", "adata_UMAP2"]] = values
                if not has_default_umap:
                    result.loc[shared, ["UMAP1", "UMAP2"]] = values
    return result


def _run_gene_set_score(adata, df_sim: pd.DataFrame, label: str, gene_set: list[str] | None) -> dict[str, Any]:
    if adata is None or not gene_set:
        return {"status": "skipped", "reason": "missing_adata_or_gene_set", "n_genes_used": 0}
    genes = [gene for gene in gene_set if gene in set(map(str, adata.var_names))]
    missing = sorted(set(gene_set) - set(genes))
    score_name = f"{_safe_component(label)}_gene_set_score"
    if not genes:
        return {"status": "skipped", "reason": "no_gene_set_genes_present", "missing_genes": missing, "n_genes_used": 0}
    sc = _scanpy()
    sc.tl.score_genes(adata, gene_list=genes, score_name=score_name)
    if score_name not in adata.obs:
        adata.obs[score_name] = np.nan
    shared = df_sim.index.intersection(adata.obs.index)
    df_sim.loc[shared, "gene_set_score"] = adata.obs.loc[shared, score_name].values
    valid = df_sim[["score", "gene_set_score"]].dropna()
    group_means = df_sim.groupby("associated")["gene_set_score"].mean()
    return {
        "status": "ok",
        "score_name": score_name,
        "n_genes_used": len(genes),
        "missing_genes": missing,
        "pearson": float(valid["score"].corr(valid["gene_set_score"], method="pearson")) if len(valid) > 1 else np.nan,
        "spearman": float(valid["score"].corr(valid["gene_set_score"], method="spearman")) if len(valid) > 1 else np.nan,
        "mean_gene_set_score_associated": float(group_means.get(True, np.nan)),
        "mean_gene_set_score_unassociated": float(group_means.get(False, np.nan)),
        "mean_gene_set_score_difference": float(group_means.get(True, np.nan) - group_means.get(False, np.nan)),
    }


def _run_deg(adata, df_sim: pd.DataFrame, output_dir: Path, min_group_cells: int) -> pd.DataFrame:
    if adata is None:
        return pd.DataFrame([{"status": "skipped", "reason": "missing_adata"}])
    counts = df_sim["associated"].value_counts()
    if counts.get(True, 0) < min_group_cells or counts.get(False, 0) < min_group_cells:
        return pd.DataFrame([{"status": "skipped", "reason": "min_group_cells", "n_true": int(counts.get(True, 0)), "n_false": int(counts.get(False, 0))}])
    shared = df_sim.index.intersection(adata.obs.index)
    adata_sub = adata[adata.obs.index.isin(shared)].copy()
    adata_sub.obs["associated"] = df_sim.loc[adata_sub.obs_names, "associated"].map({True: "True", False: "False"}).astype("category")
    sc = _scanpy()
    sc.settings.figdir = str(output_dir)
    sc.tl.rank_genes_groups(adata_sub, groupby="associated", reference="False", method="wilcoxon")
    deg_df = sc.get.rank_genes_groups_df(adata_sub, group="True")
    if deg_df.empty:
        return pd.DataFrame([{"status": "ok", "n_genes": 0}])
    deg_df.to_csv(output_dir / "deg_rank_genes_groups.csv", index=False)
    top_genes = deg_df.head(10)["names"].astype(str).tolist() if "names" in deg_df else []
    if top_genes:
        try:
            sc.pl.rank_genes_groups_heatmap(adata_sub, groups=["True", "False"], var_names=top_genes[:5], dendrogram=False, show=False, save="_associated.pdf")
        except Exception as error:
            print(f"Skipping DEG heatmap: {type(error).__name__}: {error}")
    return pd.DataFrame([{"status": "ok", "n_genes": int(len(deg_df)), "top_genes": ";".join(top_genes)}])


def _supervised_score_summary(df_sim: pd.DataFrame, *, cell_type: str) -> dict[str, Any]:
    if "ground_truth_positive" not in df_sim:
        return {}
    valid = df_sim[["score", "ground_truth_positive"]].dropna().copy()
    if valid.empty or valid["ground_truth_positive"].nunique() < 2:
        return {
            "cell_type": cell_type,
            "supervised_status": "skipped",
            "reason": "missing_two_ground_truth_classes",
        }

    truth = valid["ground_truth_positive"].astype(bool).to_numpy()
    scores = valid["score"].astype(float).to_numpy()
    positives = scores[truth]
    negatives = scores[~truth]
    fpr, tpr, _thresholds = _roc_curve_numpy(truth.astype(int), scores)
    try:
        from scipy.stats import mannwhitneyu

        mw_stat, mw_p = mannwhitneyu(positives, negatives, alternative="two-sided")
    except Exception:
        mw_stat, mw_p = np.nan, np.nan

    return {
        "cell_type": cell_type,
        "supervised_status": "ok",
        "n_ground_truth_positive": int(truth.sum()),
        "n_ground_truth_negative": int((~truth).sum()),
        "mean_sim_ground_truth_positive": float(np.mean(positives)),
        "mean_sim_ground_truth_negative": float(np.mean(negatives)),
        "mean_diff": float(np.mean(positives) - np.mean(negatives)),
        "mw_stat": float(mw_stat),
        "mw_p": float(mw_p),
        "roc_auc": float(np.trapezoid(tpr, fpr)),
    }


def _assign_donor_level_youden_condition(
    df_sim: pd.DataFrame,
    adata: Any,
    *,
    subject_column: str,
    score_column: str = "score",
    ground_truth_column: str = "ground_truth_positive",
    output_column: str = "donor_model_associated",
    aggregation: str = "median",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if adata is None:
        return df_sim, pd.DataFrame(), {"status": "skipped", "reason": "missing_adata"}
    if subject_column not in adata.obs:
        return (
            df_sim,
            pd.DataFrame(),
            {"status": "skipped", "reason": "missing_subject_column", "subject_column": subject_column},
        )
    missing = {score_column, ground_truth_column}.difference(df_sim.columns)
    if missing:
        return (
            df_sim,
            pd.DataFrame(),
            {"status": "skipped", "reason": "missing_columns", "missing_columns": ";".join(sorted(missing))},
        )

    shared = df_sim.index.intersection(adata.obs.index)
    if len(shared) == 0:
        return df_sim, pd.DataFrame(), {"status": "skipped", "reason": "no_shared_cells"}

    work = df_sim.loc[shared, [score_column, ground_truth_column]].copy()
    work["subject"] = adata.obs.loc[shared, subject_column].astype(str).values
    work[score_column] = pd.to_numeric(work[score_column], errors="coerce")
    work[ground_truth_column] = work[ground_truth_column].astype(bool)
    work = work.dropna(subset=[score_column, "subject"]).copy()
    if work.empty:
        return df_sim, pd.DataFrame(), {"status": "skipped", "reason": "empty_after_alignment"}

    grouped = work.groupby("subject", sort=True)
    if aggregation == "mean":
        donor_score = grouped[score_column].mean()
    elif aggregation == "median":
        donor_score = grouped[score_column].median()
    else:
        raise ValueError(f"Unknown donor score aggregation: {aggregation}")

    truth_counts = grouped[ground_truth_column].agg(
        n_cells="size",
        n_ground_truth_positive="sum",
        n_ground_truth_classes="nunique",
    )
    audit = truth_counts.join(donor_score.rename("donor_score")).reset_index()
    audit["n_ground_truth_negative"] = audit["n_cells"] - audit["n_ground_truth_positive"]
    audit["included_for_youden"] = audit["n_ground_truth_classes"] == 1
    audit["exclusion_reason"] = np.where(
        audit["included_for_youden"],
        "",
        "mixed_ground_truth_labels",
    )
    audit["donor_ground_truth_positive"] = audit["n_ground_truth_positive"] > 0

    included = audit[audit["included_for_youden"]].copy()
    if included.empty:
        return (
            df_sim,
            audit,
            {"status": "skipped", "reason": "no_single_condition_subjects", "donor_score_aggregation": aggregation},
        )
    if included["donor_ground_truth_positive"].nunique() < 2:
        return (
            df_sim,
            audit,
            {
                "status": "skipped",
                "reason": "missing_two_ground_truth_classes",
                "donor_score_aggregation": aggregation,
                "n_donors": int(len(included)),
                "n_donors_positive": int(included["donor_ground_truth_positive"].sum()),
                "n_donors_negative": int((~included["donor_ground_truth_positive"]).sum()),
            },
        )

    truth = included["donor_ground_truth_positive"].astype(bool).to_numpy()
    scores = included["donor_score"].astype(float).to_numpy()
    fpr, tpr, thresholds = _roc_curve_numpy(truth.astype(int), scores)
    finite_mask = np.isfinite(thresholds)
    candidate_indices = np.where(finite_mask)[0]
    if len(candidate_indices) == 0:
        return (
            df_sim,
            audit,
            {"status": "skipped", "reason": "no_finite_youden_threshold", "donor_score_aggregation": aggregation},
        )
    youden = tpr - fpr
    best_idx = int(candidate_indices[np.argmax(youden[candidate_indices])])
    threshold = float(thresholds[best_idx])
    donor_conditions = included.set_index("subject")["donor_score"] >= threshold
    audit["donor_youden_threshold"] = threshold
    audit["donor_model_associated"] = audit["subject"].map(donor_conditions)

    assigned = df_sim.copy()
    subject_by_cell = adata.obs.loc[shared, subject_column].astype(str)
    assigned.loc[shared, output_column] = subject_by_cell.map(donor_conditions)
    assigned = assigned.dropna(subset=[output_column]).copy()
    assigned[output_column] = assigned[output_column].astype(bool)

    return (
        assigned,
        audit,
        {
            "status": "ok",
            "donor_score_aggregation": aggregation,
            "donor_youden_threshold": threshold,
            "donor_roc_auc": float(np.trapezoid(tpr, fpr)),
            "donor_youden_j": float(youden[best_idx]),
            "n_donors": int(len(included)),
            "n_donors_positive": int(truth.sum()),
            "n_donors_negative": int((~truth).sum()),
            "n_model_associated_donors": int(donor_conditions.sum()),
            "n_model_unassociated_donors": int((~donor_conditions).sum()),
            "n_excluded_mixed_ground_truth_donors": int((~audit["included_for_youden"]).sum()),
        },
    )


def _plot_ground_truth_similarity_histogram(df_sim: pd.DataFrame, output_path: Path, *, bins: int) -> None:
    if "ground_truth_positive" not in df_sim:
        return
    valid = df_sim[["score", "ground_truth_positive"]].dropna().copy()
    if valid.empty or valid["ground_truth_positive"].nunique() < 2:
        return

    valid["ground_truth_status"] = valid["ground_truth_positive"].map(
        {True: "Ground truth positive", False: "Ground truth negative"}
    )
    fig, ax = plt.subplots(figsize=(2.8, 1.9))
    sns.histplot(
        data=valid,
        x="score",
        hue="ground_truth_status",
        bins=_histogram_edges(valid["score"], bins),
        common_norm=False,
        stat="density",
        element="step",
        fill=True,
        alpha=0.35,
        palette={
            "Ground truth positive": TRUE_FALSE_COLORS[True],
            "Ground truth negative": TRUE_FALSE_COLORS[False],
        },
        ax=ax,
    )
    ax.set_xlabel("Functionality similarity score", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Density", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    sns.despine(ax=ax)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_supervised_celltype_summary(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty or not {"cell_type", "mean_diff", "mw_p"}.issubset(summary.columns):
        return
    plot_df = summary[summary["supervised_status"] == "ok"].copy()
    if plot_df.empty:
        return
    plot_df["-log10p"] = -np.log10(plot_df["mw_p"].astype(float) + 1e-300)
    plot_df["cell_count"] = (
        plot_df["n_ground_truth_positive"].astype(float)
        + plot_df["n_ground_truth_negative"].astype(float)
    )
    significance_cutoff = -np.log10(0.05)
    plot_df["significant"] = plot_df["mw_p"].astype(float) < 0.05

    fig, ax = plt.subplots(figsize=(3.1, 2.2))
    size_values = 35 + 1140 * plot_df["cell_count"] / max(float(plot_df["cell_count"].max()), 1.0)
    for significant, color, alpha in [
        (False, DISTRIBUTION_DIFFERENCE_COLORS["nonsignificant"], 0.68),
        (True, DISTRIBUTION_DIFFERENCE_COLORS["significant"], 0.86),
    ]:
        mask = plot_df["significant"] == significant
        if not mask.any():
            continue
        ax.scatter(
            plot_df.loc[mask, "mean_diff"],
            plot_df.loc[mask, "-log10p"],
            s=size_values.loc[mask],
            color=color,
            edgecolor="0.15",
            linewidth=0.25,
            alpha=alpha,
            zorder=3 if significant else 2,
        )
    for _, row in plot_df.iterrows():
        ax.text(
            float(row["mean_diff"]),
            float(row["-log10p"]),
            str(row["cell_type"]),
            fontsize=TICK_LABEL_SIZE,
            ha="left",
            va="bottom",
        )
    ax.axvline(0, color="0.65", linewidth=0.55)
    ax.axhline(significance_cutoff, color="0.35", linewidth=0.65, linestyle="--")
    ax.set_xlabel("Mean similarity difference", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("-log10(p-value)", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    sns.despine(ax=ax)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def _run_ground_truth_deg_comparison(
    adata: Any,
    df_sim: pd.DataFrame,
    output_dir: Path,
    *,
    min_group_cells: int,
) -> pd.DataFrame:
    if adata is None:
        return pd.DataFrame([{"status": "skipped", "reason": "missing_adata"}])
    if "ground_truth_positive" not in df_sim or "associated" not in df_sim:
        return pd.DataFrame([{"status": "skipped", "reason": "missing_ground_truth_or_associated"}])

    shared = df_sim.index.intersection(adata.obs.index)
    if len(shared) == 0:
        return pd.DataFrame([{"status": "skipped", "reason": "no_shared_cells"}])
    work = df_sim.loc[shared, ["associated", "ground_truth_positive"]].copy()
    associated_counts = work["associated"].value_counts()
    ground_truth_counts = work["ground_truth_positive"].value_counts()
    if (
        associated_counts.get(True, 0) < min_group_cells
        or associated_counts.get(False, 0) < min_group_cells
        or ground_truth_counts.get(True, 0) < min_group_cells
        or ground_truth_counts.get(False, 0) < min_group_cells
    ):
        return pd.DataFrame(
            [
                {
                    "status": "skipped",
                    "reason": "min_group_cells",
                    "n_associated_true": int(associated_counts.get(True, 0)),
                    "n_associated_false": int(associated_counts.get(False, 0)),
                    "n_ground_truth_true": int(ground_truth_counts.get(True, 0)),
                    "n_ground_truth_false": int(ground_truth_counts.get(False, 0)),
                }
            ]
        )

    sc = _scanpy()
    adata_sub = adata[adata.obs.index.isin(shared)].copy()
    adata_sub.obs["associated"] = work.loc[adata_sub.obs_names, "associated"].map({True: "True", False: "False"}).astype("category")
    adata_sub.obs["ground_truth_positive"] = work.loc[adata_sub.obs_names, "ground_truth_positive"].map({True: "True", False: "False"}).astype("category")

    sc.tl.rank_genes_groups(adata_sub, groupby="associated", reference="False", method="wilcoxon")
    associated_deg = sc.get.rank_genes_groups_df(adata_sub, group="True")
    sc.tl.rank_genes_groups(adata_sub, groupby="ground_truth_positive", reference="False", method="wilcoxon")
    ground_truth_deg = sc.get.rank_genes_groups_df(adata_sub, group="True")
    associated_deg.to_csv(output_dir / "associated_deg_rank_genes_groups.csv", index=False)
    ground_truth_deg.to_csv(output_dir / "ground_truth_deg_rank_genes_groups.csv", index=False)
    if not ground_truth_deg.empty and "names" in ground_truth_deg:
        ground_truth_top = ground_truth_deg.head(5)["names"].astype(str).tolist()
        if ground_truth_top:
            try:
                sc.settings.figdir = str(output_dir)
                sc.pl.rank_genes_groups_heatmap(
                    adata_sub,
                    groups=["True", "False"],
                    var_names=ground_truth_top,
                    dendrogram=False,
                    show=False,
                    save="_ground_truth_positive.pdf",
                )
            except Exception as error:
                print(f"Skipping ground-truth DEG heatmap: {type(error).__name__}: {error}")

    if associated_deg.empty or ground_truth_deg.empty:
        return pd.DataFrame([{"status": "ok", "reason": "empty_deg"}])
    required = {"names", "logfoldchanges", "pvals_adj"}
    if not required.issubset(associated_deg.columns) or not required.issubset(ground_truth_deg.columns):
        return pd.DataFrame([{"status": "ok", "reason": "missing_deg_columns"}])

    associated_top = associated_deg.head(5)["names"].astype(str).tolist()
    ground_truth_top = ground_truth_deg.head(5)["names"].astype(str).tolist()
    overlap_genes = list(dict.fromkeys([gene for gene in associated_top if gene in set(ground_truth_top)]))
    if not overlap_genes:
        pd.DataFrame(
            columns=[
                "names",
                "logfoldchanges_associated",
                "pvals_adj_associated",
                "logfoldchanges_ground_truth",
                "pvals_adj_ground_truth",
            ]
        ).to_csv(output_dir / "ground_truth_deg_comparison.csv", index=False)
        return pd.DataFrame(
            [
                {
                    "status": "ok",
                    "reason": "no_top5_overlap_genes",
                    "n_associated_deg_genes": int(len(associated_deg)),
                    "n_ground_truth_deg_genes": int(len(ground_truth_deg)),
                    "associated_top5_genes": ";".join(associated_top),
                    "ground_truth_top5_genes": ";".join(ground_truth_top),
                    "n_compared_genes": 0,
                    "compared_genes": "",
                }
            ]
        )

    comparison = associated_deg[associated_deg["names"].astype(str).isin(overlap_genes)][
        ["names", "logfoldchanges", "pvals_adj"]
    ].merge(
        ground_truth_deg[ground_truth_deg["names"].astype(str).isin(overlap_genes)][
            ["names", "logfoldchanges", "pvals_adj"]
        ],
        on="names",
        suffixes=("_associated", "_ground_truth"),
    )
    comparison.to_csv(output_dir / "ground_truth_deg_comparison.csv", index=False)
    _plot_deg_comparison_bars(comparison, output_dir)
    return pd.DataFrame(
        [
            {
                "status": "ok",
                "n_associated_deg_genes": int(len(associated_deg)),
                "n_ground_truth_deg_genes": int(len(ground_truth_deg)),
                "associated_top5_genes": ";".join(associated_top),
                "ground_truth_top5_genes": ";".join(ground_truth_top),
                "n_compared_genes": int(len(comparison)),
                "compared_genes": ";".join(comparison["names"].astype(str).tolist()),
            }
        ]
    )


def _matrix_to_dense(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray())
    return np.asarray(matrix)


def _subject_level_deseq2(
    adata: Any,
    df_sim: pd.DataFrame,
    output_dir: Path,
    *,
    subject_column: str,
    layer: str,
    min_subjects_per_group: int,
    condition_column: str = "ground_truth_positive",
    output_prefix: str = "",
    split_mixed_subjects: bool = False,
) -> pd.DataFrame:
    if adata is None:
        return pd.DataFrame([{"status": "skipped", "reason": "missing_adata"}])
    if condition_column not in df_sim:
        return pd.DataFrame([{"status": "skipped", "reason": "missing_condition", "condition_column": condition_column}])
    if subject_column not in adata.obs:
        return pd.DataFrame([{"status": "skipped", "reason": "missing_subject_column", "subject_column": subject_column}])

    shared = df_sim.index.intersection(adata.obs.index)
    if len(shared) == 0:
        return pd.DataFrame([{"status": "skipped", "reason": "no_shared_cells"}])
    adata_sub = adata[adata.obs.index.isin(shared)].copy()
    truth = df_sim.loc[adata_sub.obs_names, condition_column].astype(bool)
    subject_values = adata_sub.obs[subject_column].astype(str)
    sample_meta = pd.DataFrame({"subject": subject_values.values, "condition": truth.map({True: "True", False: "False"}).values}, index=adata_sub.obs_names)
    if not split_mixed_subjects:
        subject_condition_counts = sample_meta.groupby("subject")["condition"].nunique()
        mixed_subjects = subject_condition_counts[subject_condition_counts > 1].index
        if len(mixed_subjects) > 0:
            sample_meta = sample_meta[~sample_meta["subject"].isin(mixed_subjects)].copy()
            adata_sub = adata_sub[adata_sub.obs_names.isin(sample_meta.index)].copy()
    else:
        subject_condition_counts = sample_meta.groupby("subject")["condition"].nunique()
        paired_subjects = subject_condition_counts[subject_condition_counts == 2].index
        sample_meta = sample_meta[sample_meta["subject"].isin(paired_subjects)].copy()
        adata_sub = adata_sub[adata_sub.obs_names.isin(sample_meta.index)].copy()
    if sample_meta.empty:
        reason = "no_paired_subjects" if split_mixed_subjects else "no_single_condition_subjects"
        return pd.DataFrame([{"status": "skipped", "reason": reason}])

    if split_mixed_subjects:
        sample_meta["pseudobulk_sample"] = sample_meta["subject"] + "__" + sample_meta["condition"]
        sample_conditions = sample_meta.drop_duplicates("pseudobulk_sample").set_index("pseudobulk_sample")["condition"]
        subject_conditions = sample_meta.drop_duplicates("subject").set_index("subject")["condition"]
        condition_subject_counts = sample_meta.drop_duplicates(["subject", "condition"]).groupby("condition")["subject"].nunique()
    else:
        sample_meta["pseudobulk_sample"] = sample_meta["subject"]
        sample_conditions = sample_meta.drop_duplicates("subject").set_index("subject")["condition"]
        subject_conditions = sample_conditions
        condition_subject_counts = sample_conditions.reset_index().groupby("condition")["subject"].nunique()
    condition_counts = condition_subject_counts
    if condition_counts.get("True", 0) < min_subjects_per_group or condition_counts.get("False", 0) < min_subjects_per_group:
        return pd.DataFrame(
            [
                {
                    "status": "skipped",
                    "reason": "min_subjects_per_group",
                    "n_subjects_true": int(condition_counts.get("True", 0)),
                    "n_subjects_false": int(condition_counts.get("False", 0)),
                    "min_subjects_per_group": int(min_subjects_per_group),
                }
            ]
        )

    if layer:
        if not hasattr(adata_sub, "layers") or layer not in adata_sub.layers:
            return pd.DataFrame([{"status": "skipped", "reason": "missing_pseudobulk_layer", "layer": layer}])
        matrix = adata_sub.layers[layer]
    else:
        matrix = adata_sub.X
    counts = _matrix_to_dense(matrix)
    if counts.ndim != 2 or counts.shape[0] != len(adata_sub.obs):
        return pd.DataFrame([{"status": "skipped", "reason": "invalid_count_matrix"}])
    counts = np.rint(counts).astype(np.int64)
    counts[counts < 0] = 0
    genes = pd.Index(adata_sub.var_names.astype(str))

    pseudobulk_rows: list[np.ndarray] = []
    pseudobulk_index: list[str] = []
    for sample_id in sample_conditions.index:
        mask = sample_meta["pseudobulk_sample"].to_numpy() == sample_id
        if not np.any(mask):
            continue
        pseudobulk_rows.append(counts[mask].sum(axis=0))
        pseudobulk_index.append(sample_id)
    if not pseudobulk_rows:
        return pd.DataFrame([{"status": "skipped", "reason": "empty_pseudobulk"}])

    counts_df = pd.DataFrame(np.vstack(pseudobulk_rows), index=pseudobulk_index, columns=genes)
    metadata = pd.DataFrame({"condition": sample_conditions.loc[pseudobulk_index].values}, index=pseudobulk_index)
    metadata["subject"] = [str(sample_id).rsplit("__", maxsplit=1)[0] for sample_id in pseudobulk_index]
    nonzero_genes = counts_df.sum(axis=0) > 0
    counts_df = counts_df.loc[:, nonzero_genes]
    if counts_df.empty:
        return pd.DataFrame([{"status": "skipped", "reason": "no_nonzero_genes"}])
    counts_df.to_csv(output_dir / f"{output_prefix}subject_level_pseudobulk_counts.csv")
    metadata.to_csv(output_dir / f"{output_prefix}subject_level_pseudobulk_metadata.csv")

    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats

        design = "~subject + condition" if split_mixed_subjects else "~condition"
        dds = DeseqDataSet(
            counts=counts_df,
            metadata=metadata,
            design=design,
            ref_level=["condition", "False"],
            quiet=True,
            n_cpus=1,
        )
        dds.deseq2()
        stats = DeseqStats(
            dds,
            contrast=["condition", "True", "False"],
            quiet=True,
            n_cpus=1,
        )
        stats.summary()
        results = stats.results_df.copy()
    except Exception as error:
        return pd.DataFrame([{"status": "skipped", "reason": "pydeseq2_error", "error": f"{type(error).__name__}: {error}"}])

    results = results.reset_index().rename(columns={"index": "gene"})
    results.to_csv(output_dir / f"{output_prefix}subject_level_deseq2_results.csv", index=False)
    return pd.DataFrame(
        [
            {
                "status": "ok",
                "n_subjects": int(len(metadata)),
                "n_unique_subjects": int(metadata["subject"].nunique()),
                "n_subjects_true": int((metadata["condition"] == "True").sum()),
                "n_subjects_false": int((metadata["condition"] == "False").sum()),
                "n_genes": int(counts_df.shape[1]),
                "counts_layer": layer or "X",
                "design": design,
                "condition_column": condition_column,
                "output_prefix": output_prefix,
                "split_mixed_subjects": bool(split_mixed_subjects),
            }
        ]
    )


def _plot_deg_comparison_bars(
    comparison: pd.DataFrame,
    output_dir: Path,
    *,
    file_prefix: str = "",
    logfc_columns: tuple[str, str] = ("logfoldchanges_associated", "logfoldchanges_ground_truth"),
    pvalue_columns: tuple[str, str] = ("pvals_adj_associated", "pvals_adj_ground_truth"),
    method_labels: tuple[str, str] = ("Model associated", "Ground truth"),
) -> None:
    if comparison.empty:
        return
    palette = DEG_COMPARISON_COLORS
    logfc_left, logfc_right = logfc_columns
    pvalue_left, pvalue_right = pvalue_columns
    method_left, method_right = method_labels

    gene_labels = comparison["names"].astype(str).tolist()
    x_positions = np.arange(len(gene_labels), dtype=float) * DEG_COMPARISON_GENE_X_SPACING
    fig_width = max(2.2, DEG_COMPARISON_GENE_X_SPACING * len(gene_labels) + 1.3)

    fig, ax = plt.subplots(figsize=(fig_width, 2.1))
    ax.bar(
        x_positions - DEG_COMPARISON_BAR_WIDTH / 2,
        comparison[logfc_left].astype(float),
        width=DEG_COMPARISON_BAR_WIDTH,
        color=palette[0],
        edgecolor="0.15",
        linewidth=0.35,
        label=method_left,
    )
    ax.bar(
        x_positions + DEG_COMPARISON_BAR_WIDTH / 2,
        comparison[logfc_right].astype(float),
        width=DEG_COMPARISON_BAR_WIDTH,
        color=palette[1],
        edgecolor="0.15",
        linewidth=0.35,
        label=method_right,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Log fold change", fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(gene_labels, rotation=45, ha="right", fontsize=TICK_LABEL_SIZE)
    for tick_label in ax.get_xticklabels():
        tick_label.set_fontstyle("italic")
    ax.tick_params(axis="x", labelrotation=45, labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    if len(x_positions):
        ax.set_xlim(min(x_positions) - 0.28, max(x_positions) + 0.28)
    ax.legend(frameon=False, fontsize=TICK_LABEL_SIZE, loc="center left", bbox_to_anchor=(1.0, 0.5))
    sns.despine(ax=ax)
    fig.savefig(output_dir / f"{file_prefix}logFC_barplot.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{file_prefix}logFC_barplot.svg", bbox_inches="tight")
    plt.close(fig)

    minus_log10_left = -np.log10(comparison[pvalue_left].astype(float) + 1e-300)
    minus_log10_right = -np.log10(comparison[pvalue_right].astype(float) + 1e-300)
    fig, ax = plt.subplots(figsize=(fig_width, 2.1))
    ax.bar(
        x_positions - DEG_COMPARISON_BAR_WIDTH / 2,
        minus_log10_left,
        width=DEG_COMPARISON_BAR_WIDTH,
        color=palette[0],
        edgecolor="0.15",
        linewidth=0.35,
        label=method_left,
    )
    ax.bar(
        x_positions + DEG_COMPARISON_BAR_WIDTH / 2,
        minus_log10_right,
        width=DEG_COMPARISON_BAR_WIDTH,
        color=palette[1],
        edgecolor="0.15",
        linewidth=0.35,
        label=method_right,
    )
    ax.set_xlabel("")
    ax.set_ylabel("-log10(p-value)", fontsize=AXIS_LABEL_SIZE)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(gene_labels, rotation=45, ha="right", fontsize=TICK_LABEL_SIZE)
    for tick_label in ax.get_xticklabels():
        tick_label.set_fontstyle("italic")
    ax.tick_params(axis="x", labelrotation=45, labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    if len(x_positions):
        ax.set_xlim(min(x_positions) - 0.28, max(x_positions) + 0.28)
    ax.legend(frameon=False, fontsize=TICK_LABEL_SIZE, loc="center left", bbox_to_anchor=(1.0, 0.5))
    sns.despine(ax=ax)
    fig.savefig(output_dir / f"{file_prefix}pvalue_barplot.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{file_prefix}pvalue_barplot.svg", bbox_inches="tight")
    plt.close(fig)


def _top_pydeseq2_genes(results: pd.DataFrame, *, top_n: int = 5) -> list[str]:
    if results.empty or "gene" not in results:
        return []
    work = results.copy()
    pvalue_column = "padj" if "padj" in work.columns else "pvalue" if "pvalue" in work.columns else None
    if pvalue_column is not None:
        work[pvalue_column] = pd.to_numeric(work[pvalue_column], errors="coerce")
        work = work.sort_values(
            by=[pvalue_column, "log2FoldChange"] if "log2FoldChange" in work.columns else [pvalue_column],
            ascending=[True, False] if "log2FoldChange" in work.columns else [True],
            key=lambda values: values.abs() if values.name == "log2FoldChange" else values,
        )
    return work["gene"].astype(str).head(top_n).tolist()


def _write_pydeseq2_deg_comparison(output_dir: Path) -> pd.DataFrame:
    associated_path = output_dir / "associated_subject_level_deseq2_results.csv"
    ground_truth_path = output_dir / "ground_truth_subject_level_deseq2_results.csv"
    if not associated_path.exists() or not ground_truth_path.exists():
        return pd.DataFrame([{"status": "skipped", "reason": "missing_deg_inputs"}])

    associated_results = pd.read_csv(associated_path)
    ground_truth_results = pd.read_csv(ground_truth_path)
    associated_pvalue_column = "padj" if "padj" in associated_results.columns else "pvalue" if "pvalue" in associated_results.columns else None
    ground_truth_pvalue_column = "padj" if "padj" in ground_truth_results.columns else "pvalue" if "pvalue" in ground_truth_results.columns else None
    if (
        not {"gene", "log2FoldChange"}.issubset(associated_results.columns)
        or not {"gene", "log2FoldChange"}.issubset(ground_truth_results.columns)
        or associated_pvalue_column is None
        or ground_truth_pvalue_column is None
    ):
        return pd.DataFrame([{"status": "skipped", "reason": "missing_deg_columns"}])

    associated_top = _top_pydeseq2_genes(associated_results, top_n=5)
    ground_truth_top = _top_pydeseq2_genes(ground_truth_results, top_n=5)
    overlap_genes = list(dict.fromkeys([gene for gene in associated_top if gene in set(ground_truth_top)]))
    comparison_genes = list(dict.fromkeys([*associated_top, *ground_truth_top]))
    if not comparison_genes:
        pd.DataFrame(
            columns=[
                "names",
                "logfoldchanges_associated_pydeseq2",
                "pvals_adj_associated_pydeseq2",
                "logfoldchanges_ground_truth_pydeseq2",
                "pvals_adj_ground_truth_pydeseq2",
            ]
        ).to_csv(output_dir / "pydeseq2_deg_comparison.csv", index=False)
        return pd.DataFrame(
            [
                {
                    "status": "ok",
                    "reason": "no_top5_genes",
                    "associated_top5_genes": ";".join(associated_top),
                    "ground_truth_top5_genes": ";".join(ground_truth_top),
                    "n_compared_genes": 0,
                    "n_top5_overlap_genes": 0,
                    "top5_overlap_genes": "",
                    "compared_genes": "",
                }
            ]
        )

    comparison = associated_results[associated_results["gene"].astype(str).isin(comparison_genes)][
        ["gene", "log2FoldChange", associated_pvalue_column]
    ].rename(
        columns={
            "gene": "names",
            "log2FoldChange": "logfoldchanges_associated_pydeseq2",
            associated_pvalue_column: "pvals_adj_associated_pydeseq2",
        }
    ).merge(
        ground_truth_results[ground_truth_results["gene"].astype(str).isin(comparison_genes)][
            ["gene", "log2FoldChange", ground_truth_pvalue_column]
        ].rename(
            columns={
                "gene": "names",
                "log2FoldChange": "logfoldchanges_ground_truth_pydeseq2",
                ground_truth_pvalue_column: "pvals_adj_ground_truth_pydeseq2",
            }
        ),
        on="names",
    )
    comparison["_plot_order"] = comparison["names"].astype(str).map(
        {gene: index for index, gene in enumerate(comparison_genes)}
    )
    comparison = comparison.sort_values("_plot_order", kind="mergesort").drop(columns="_plot_order")
    comparison.to_csv(output_dir / "pydeseq2_deg_comparison.csv", index=False)
    _plot_deg_comparison_bars(
        comparison,
        output_dir,
        file_prefix="pydeseq2_",
        logfc_columns=("logfoldchanges_associated_pydeseq2", "logfoldchanges_ground_truth_pydeseq2"),
        pvalue_columns=("pvals_adj_associated_pydeseq2", "pvals_adj_ground_truth_pydeseq2"),
        method_labels=("Model-associated subject-level DESeq2", "Ground-truth subject-level DESeq2"),
    )
    return pd.DataFrame(
        [
            {
                "status": "ok",
                "associated_top5_genes": ";".join(associated_top),
                "ground_truth_top5_genes": ";".join(ground_truth_top),
                "n_compared_genes": int(len(comparison)),
                "n_top5_overlap_genes": int(len(overlap_genes)),
                "top5_overlap_genes": ";".join(overlap_genes),
                "compared_genes": ";".join(comparison["names"].astype(str).tolist()),
            }
        ]
    )


def _json_safe_config(config: FunctionalityDownstreamConfig) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return [convert(item) for item in value]
        if isinstance(value, tuple):
            return [convert(item) for item in value]
        return value

    return convert(asdict(config))


def _analysis_groups(work: pd.DataFrame, config: FunctionalityDownstreamConfig) -> list[tuple[str, pd.DataFrame]]:
    if config.assessment_scope == "all_cells":
        return [("all_cells", work.copy())]
    if config.assessment_scope != "per_cell_type":
        raise ValueError(f"Unknown assessment_scope: {config.assessment_scope}")

    cell_types = sorted(work[config.annotation_column].dropna().astype(str).unique())
    return [
        (cell_type, work[work[config.annotation_column].astype(str) == cell_type].copy())
        for cell_type in cell_types
    ]


def run_functionality_downstream_assessment(
    embeddings_dict: dict[str, Any],
    *,
    eval_data: Any | None = None,
    config: FunctionalityDownstreamConfig,
) -> dict[str, Path]:
    if config.plot:
        _set_annotation_plot_style()
    run_dir = _unique_dir(Path(config.output_dir) / _safe_component(config.timestamp or _timestamp_now()))
    run_dir.mkdir(parents=True, exist_ok=False)
    adata = _load_adata(eval_data, config)
    summary_rows: list[dict[str, Any]] = []
    gene_set_scatter_frames: list[pd.DataFrame] = []
    rank_tail_metric_frames: list[pd.DataFrame] = []
    rank_tail_summary_rows: list[dict[str, Any]] = []

    for model_name, model_data in embeddings_dict.items():
        model_dir = run_dir / _safe_component(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        for dataset_name, dataset_meta in model_data.items():
            loaded = _load_artifacts(dataset_meta, config.annotation_column)
            artifacts = loaded["artifacts"]
            if "df_cells" not in artifacts:
                continue
            cell_df = artifacts["df_cells"]["dataframe"].copy()
            if config.annotation_column not in cell_df.columns:
                raise ValueError(f"Cell embeddings are missing {config.annotation_column!r}.")
            embedding_cols = _embedding_columns(cell_df, {config.annotation_column})
            function_df = _functionality_embeddings(
                artifacts,
                config.functionality_labels,
                model_name=_resolve_encoding_model_name(config, model_name),
                allow_live_encoding=config.allow_live_functionality_encoding,
            )
            function_cols = _embedding_columns(function_df, {"functionality", "data"})
            function_df = function_df.set_index("functionality")

            for label in config.functionality_labels:
                label_dir = model_dir / _safe_component(label)
                label_dir.mkdir(parents=True, exist_ok=True)
                supervised_rows_for_label: list[dict[str, Any]] = []
                label_vector = function_df.loc[[label], function_cols].to_numpy(dtype=np.float32)
                scores = _similarity(cell_df[embedding_cols].to_numpy(dtype=np.float32), label_vector, config.similarity_metric).ravel()
                work = cell_df[[config.annotation_column]].copy()
                work["score"] = scores
                work = _attach_umap_coordinates(work, cell_artifact=artifacts["df_cells"], adata=adata)
                if adata is not None and config.ground_truth_column and config.ground_truth_column in adata.obs:
                    shared = work.index.intersection(adata.obs.index)
                    work.loc[shared, config.ground_truth_column] = adata.obs.loc[shared, config.ground_truth_column].values
                    work.loc[shared, "ground_truth_positive"] = adata.obs.loc[shared, config.ground_truth_column].map(
                        lambda value: _is_positive(value, config.positive_values)
                    ).astype(bool)

                cell_types = sorted(work[config.annotation_column].dropna().astype(str).unique())
                if config.cell_type_filter is not None:
                    keep = set(config.cell_type_filter)
                    work = work[work[config.annotation_column].astype(str).isin(keep)].copy()
                    cell_types = [cell_type for cell_type in cell_types if cell_type in keep]

                for cell_type, df_sim in _analysis_groups(work, config):
                    celltype_dir = label_dir / _safe_component(cell_type)
                    celltype_dir.mkdir(parents=True, exist_ok=True)
                    ground_truth = df_sim["ground_truth_positive"].to_numpy(dtype=bool) if "ground_truth_positive" in df_sim else None
                    cutoff = compute_cutoff(
                        df_sim["score"],
                        method=config.cutoff_method,
                        functionality_label=label,
                        ground_truth=ground_truth,
                        manual_cutoffs=config.manual_cutoffs,
                    )
                    df_sim["associated"] = df_sim["score"] >= cutoff.cutoff

                    adata_sub = None
                    if adata is not None:
                        shared = df_sim.index.intersection(adata.obs.index)
                        adata_sub = adata[adata.obs.index.isin(shared)].copy()
                    supervised_summary = _supervised_score_summary(df_sim, cell_type=cell_type)
                    if supervised_summary:
                        supervised_summary.update(
                            {
                                "model_name": model_name,
                                "dataset_name": dataset_name,
                                "functionality": label,
                            }
                        )
                        supervised_rows_for_label.append(supervised_summary)
                    gene_summary = _run_gene_set_score(adata_sub, df_sim, label, config.gene_sets.get(label))
                    cutoff_sensitivity = (
                        _cutoff_sensitivity_table(
                            df_sim,
                            cutoff=cutoff.cutoff,
                            window_fraction=config.cutoff_sensitivity_window_fraction,
                            steps=config.cutoff_sensitivity_steps,
                            threshold_min=config.cutoff_sensitivity_threshold_min,
                            threshold_max=config.cutoff_sensitivity_threshold_max,
                        )
                        if config.cutoff_sensitivity
                        else pd.DataFrame()
                    )
                    deg_summary = _run_deg(adata_sub, df_sim, celltype_dir, config.min_group_cells)
                    ground_truth_deg_summary = (
                        _run_ground_truth_deg_comparison(
                            adata_sub,
                            df_sim,
                            celltype_dir,
                            min_group_cells=config.min_group_cells,
                        )
                        if config.supervised_downstream_plots and "ground_truth_positive" in df_sim
                        else pd.DataFrame([{"status": "skipped", "reason": "missing_ground_truth"}])
                    )
                    ground_truth_subject_level_deg_summary = (
                        _subject_level_deseq2(
                            adata_sub,
                            df_sim,
                            celltype_dir,
                            subject_column=config.subject_column,
                            layer=config.pseudobulk_layer,
                            min_subjects_per_group=config.min_subjects_per_group,
                            condition_column="ground_truth_positive",
                            output_prefix="ground_truth_",
                            split_mixed_subjects=False,
                        )
                        if config.subject_level_deg and "ground_truth_positive" in df_sim
                        else pd.DataFrame([{"status": "skipped", "reason": "disabled"}])
                    )
                    associated_subject_level_deg_summary = (
                        pd.DataFrame([{"status": "skipped", "reason": "initialized"}])
                        if config.subject_level_deg and "associated" in df_sim
                        else pd.DataFrame([{"status": "skipped", "reason": "disabled"}])
                    )
                    associated_donor_audit = pd.DataFrame()
                    associated_donor_summary_values: dict[str, Any] = {}
                    if config.subject_level_deg and "associated" in df_sim:
                        (
                            donor_condition_df_sim,
                            associated_donor_audit,
                            associated_donor_summary_values,
                        ) = _assign_donor_level_youden_condition(
                            df_sim,
                            adata_sub,
                            subject_column=config.subject_column,
                            score_column="score",
                            ground_truth_column="ground_truth_positive",
                            output_column="donor_model_associated",
                        )
                        if associated_donor_summary_values.get("status") == "ok":
                            associated_subject_level_deg_summary = _subject_level_deseq2(
                                adata_sub,
                                donor_condition_df_sim,
                                celltype_dir,
                                subject_column=config.subject_column,
                                layer=config.pseudobulk_layer,
                                min_subjects_per_group=config.min_subjects_per_group,
                                condition_column="donor_model_associated",
                                output_prefix="associated_",
                                split_mixed_subjects=False,
                            )
                            associated_subject_level_deg_summary = associated_subject_level_deg_summary.assign(
                                donor_condition_strategy="youden_j_median_score",
                                donor_youden_threshold=associated_donor_summary_values.get("donor_youden_threshold"),
                                donor_roc_auc=associated_donor_summary_values.get("donor_roc_auc"),
                                donor_youden_j=associated_donor_summary_values.get("donor_youden_j"),
                            )
                        else:
                            associated_subject_level_deg_summary = pd.DataFrame(
                                [
                                    {
                                        "status": "skipped",
                                        "reason": associated_donor_summary_values.get("reason", "donor_youden_condition_not_ok"),
                                        "donor_condition_strategy": "youden_j_median_score",
                                        **associated_donor_summary_values,
                                    }
                                ]
                            )
                    pydeseq2_deg_comparison_summary = (
                        _write_pydeseq2_deg_comparison(celltype_dir)
                        if not ground_truth_subject_level_deg_summary.empty
                        and str(ground_truth_subject_level_deg_summary.loc[0, "status"]) == "ok"
                        and not associated_subject_level_deg_summary.empty
                        and str(associated_subject_level_deg_summary.loc[0, "status"]) == "ok"
                        else pd.DataFrame([{"status": "skipped", "reason": "subject_level_deseq2_not_ok"}])
                    )
                    top_deg_sensitivity = _top_deg_cutoff_sensitivity_table(
                        adata_sub,
                        df_sim,
                        deg_summary,
                        cutoff_sensitivity,
                        top_n=7,
                    )
                    gene_cutoff_summary = _gene_cutoff_sensitivity_summary(
                        adata_sub,
                        df_sim,
                        cutoff_sensitivity,
                        cutoff=cutoff.cutoff,
                    )
                    gene_cutoff_top_increasing = gene_cutoff_summary.head(5).copy()
                    rank_tail_table = pd.DataFrame()
                    rank_tail_summary = pd.DataFrame()
                    rank_decile_overlap = pd.DataFrame()
                    rank_cell_type_tail = pd.DataFrame()
                    if (
                        config.rank_diagnostics
                        and config.assessment_scope == "all_cells"
                        and "gene_set_score" in df_sim.columns
                    ):
                        (
                            rank_tail_table,
                            rank_summary_values,
                            rank_decile_overlap,
                            rank_cell_type_tail,
                            ranked_gene_set_scores,
                        ) = _rank_diagnostic_outputs(
                            df_sim,
                            annotation_column=config.annotation_column,
                            top_fractions=tuple(config.rank_top_fractions),
                            target_gene_set_fraction=config.rank_target_gene_set_fraction,
                        )
                        if not ranked_gene_set_scores.empty:
                            shared_rank_index = df_sim.index.intersection(ranked_gene_set_scores.index)
                            df_sim.loc[shared_rank_index, "similarity_score_rank"] = ranked_gene_set_scores.loc[
                                shared_rank_index,
                                "embedding_percentile_rank",
                            ].to_numpy()
                            df_sim.loc[shared_rank_index, "gene_set_score_rank"] = ranked_gene_set_scores.loc[
                                shared_rank_index,
                                "gene_set_percentile_rank",
                            ].to_numpy()
                        rank_tail_summary_row = {
                            "model_name": model_name,
                            "dataset_name": dataset_name,
                            "functionality": label,
                            "cell_type": cell_type,
                            **rank_summary_values,
                        }
                        rank_tail_summary = pd.DataFrame([rank_tail_summary_row])
                        if not rank_tail_table.empty:
                            rank_tail_metric_frames.append(
                                rank_tail_table.assign(
                                    model_name=model_name,
                                    dataset_name=dataset_name,
                                    functionality=label,
                                    cell_type=cell_type,
                                )
                            )
                        rank_tail_summary_rows.append(rank_tail_summary_row)
                    if "gene_set_score" in df_sim.columns:
                        gene_set_scatter_frames.append(
                            df_sim.assign(
                                model_name=model_name,
                                dataset_name=dataset_name,
                                functionality=label,
                                cell_type=cell_type,
                            )
                        )

                    df_sim.to_csv(celltype_dir / "df_sim.csv", index=True)
                    with (celltype_dir / "cutoff_metadata.json").open("w", encoding="utf-8") as handle:
                        json.dump({"cutoff": cutoff.cutoff, "method": cutoff.method, **cutoff.metadata}, handle, indent=2, sort_keys=True)
                    pd.DataFrame([{"functionality": label, "cell_type": cell_type, "cutoff": cutoff.cutoff, "method": cutoff.method}]).to_csv(
                        celltype_dir / "threshold_summary.csv",
                        index=False,
                    )
                    pd.DataFrame([gene_summary]).to_csv(celltype_dir / "gene_set_score_summary.csv", index=False)
                    cutoff_sensitivity.to_csv(celltype_dir / "cutoff_sensitivity.csv", index=False)
                    top_deg_sensitivity.to_csv(celltype_dir / "top_deg_cutoff_sensitivity.csv", index=False)
                    gene_cutoff_summary.to_csv(celltype_dir / "gene_cutoff_sensitivity_summary.csv", index=False)
                    gene_cutoff_top_increasing.to_csv(celltype_dir / "gene_cutoff_sensitivity_top_increasing.csv", index=False)
                    if not rank_tail_table.empty:
                        rank_tail_table.to_csv(celltype_dir / "rank_tail_enrichment.csv", index=False)
                        rank_tail_summary.to_csv(celltype_dir / "rank_tail_summary.csv", index=False)
                        rank_decile_overlap.to_csv(celltype_dir / "rank_decile_overlap.csv", index=False)
                        rank_cell_type_tail.to_csv(celltype_dir / "rank_cell_type_tail_summary.csv", index=False)
                        if config.plot:
                            _plot_rank_decile_overlap_heatmap(
                                rank_decile_overlap,
                                celltype_dir / "rank_decile_overlap_heatmap.pdf",
                            )
                    deg_summary.to_csv(celltype_dir / "deg_summary.csv", index=False)
                    ground_truth_deg_summary.to_csv(celltype_dir / "ground_truth_deg_summary.csv", index=False)
                    ground_truth_subject_level_deg_summary.to_csv(
                        celltype_dir / "ground_truth_subject_level_deseq2_summary.csv",
                        index=False,
                    )
                    associated_subject_level_deg_summary.to_csv(
                        celltype_dir / "associated_subject_level_deseq2_summary.csv",
                        index=False,
                    )
                    associated_donor_audit.to_csv(
                        celltype_dir / "associated_donor_youden_assignment.csv",
                        index=False,
                    )
                    with (celltype_dir / "associated_donor_youden_assignment_summary.json").open(
                        "w",
                        encoding="utf-8",
                    ) as handle:
                        json.dump(associated_donor_summary_values, handle, indent=2, sort_keys=True)
                    ground_truth_subject_level_deg_summary.to_csv(celltype_dir / "subject_level_deseq2_summary.csv", index=False)
                    pydeseq2_deg_comparison_summary.to_csv(
                        celltype_dir / "pydeseq2_deg_comparison_summary.csv",
                        index=False,
                    )
                    if config.plot:
                        _plot_histogram(df_sim, cutoff.cutoff, celltype_dir / "score_histogram.pdf", config.bins)
                        if config.supervised_downstream_plots and "ground_truth_positive" in df_sim:
                            _plot_ground_truth_similarity_histogram(
                                df_sim,
                                celltype_dir / "score_histogram_by_ground_truth.pdf",
                                bins=config.bins,
                            )
                        if config.assessment_scope == "all_cells":
                            _plot_histogram_by_cell_type(
                                df_sim,
                                annotation_column=config.annotation_column,
                                cutoff=cutoff.cutoff,
                                output_path=celltype_dir / "score_histogram_by_cell_type.pdf",
                                bins=config.bins,
                                palette_name=config.categorical_palette_name,
                            )
                            _plot_histogram_by_cell_type(
                                df_sim,
                                annotation_column=config.annotation_column,
                                cutoff=cutoff.cutoff,
                                output_path=celltype_dir / "score_histogram_by_cell_type_no_cutoff.pdf",
                                bins=config.bins,
                                palette_name=config.categorical_palette_name,
                                show_cutoff=False,
                            )
                        _plot_scatter(df_sim, celltype_dir / "model_score_vs_gene_set_score.pdf")
                        _plot_cutoff_sensitivity(
                            cutoff_sensitivity,
                            celltype_dir / "cutoff_sensitivity.pdf",
                            cutoff=cutoff.cutoff,
                        )
                        _plot_top_deg_cutoff_sensitivity(
                            top_deg_sensitivity,
                            celltype_dir / "top_deg_cutoff_sensitivity.pdf",
                            cutoff=cutoff.cutoff,
                        )
                        _plot_gene_cutoff_sensitivity_top_increasing(
                            gene_cutoff_top_increasing,
                            celltype_dir / "gene_cutoff_sensitivity_top_increasing.pdf",
                            top_n=5,
                        )
                        _plot_umap(df_sim, celltype_dir / "umap_functionality_score.pdf", "score", continuous=True)
                        _plot_umap(df_sim, celltype_dir / "umap_associated.pdf", "associated", continuous=False)
                        _plot_umap(df_sim, celltype_dir / "umap_gene_set_score.pdf", "gene_set_score", continuous=True)
                        if (
                            config.rank_diagnostics
                            and config.assessment_scope == "all_cells"
                            and {"similarity_score_rank", "gene_set_score_rank"}.issubset(df_sim.columns)
                        ):
                            for coord_label, x_column, y_column in (
                                ("model", "model_UMAP1", "model_UMAP2"),
                                ("adata", "adata_UMAP1", "adata_UMAP2"),
                            ):
                                _plot_umap(
                                    df_sim,
                                    celltype_dir / f"umap_{coord_label}_similarity_score_rank.pdf",
                                    "similarity_score_rank",
                                    continuous=True,
                                    x_column=x_column,
                                    y_column=y_column,
                                )
                                _plot_umap(
                                    df_sim,
                                    celltype_dir / f"umap_{coord_label}_gene_set_score_rank.pdf",
                                    "gene_set_score_rank",
                                    continuous=True,
                                    x_column=x_column,
                                    y_column=y_column,
                                )
                        if config.supervised_downstream_plots and "ground_truth_positive" in df_sim:
                            _plot_umap(
                                df_sim,
                                celltype_dir / "umap_ground_truth_positive.pdf",
                                "ground_truth_positive",
                                continuous=False,
                            )
                            _plot_umap(
                                df_sim,
                                celltype_dir / "umap_cell_type.pdf",
                                config.annotation_column,
                                continuous=False,
                            )
                    summary_rows.append(
                        {
                            "model_name": model_name,
                            "dataset_name": dataset_name,
                            "functionality": label,
                            "cell_type": cell_type,
                            "assessment_scope": config.assessment_scope,
                            "cutoff": cutoff.cutoff,
                            "cutoff_method": cutoff.method,
                            "n_cells": int(len(df_sim)),
                            "n_associated": int(df_sim["associated"].sum()),
                            "gene_set_status": gene_summary.get("status"),
                            "deg_status": deg_summary.loc[0, "status"] if not deg_summary.empty and "status" in deg_summary else np.nan,
                            "ground_truth_deg_status": ground_truth_deg_summary.loc[0, "status"]
                            if not ground_truth_deg_summary.empty and "status" in ground_truth_deg_summary
                            else np.nan,
                            "ground_truth_subject_level_deg_status": ground_truth_subject_level_deg_summary.loc[0, "status"]
                            if not ground_truth_subject_level_deg_summary.empty and "status" in ground_truth_subject_level_deg_summary
                            else np.nan,
                            "associated_subject_level_deg_status": associated_subject_level_deg_summary.loc[0, "status"]
                            if not associated_subject_level_deg_summary.empty and "status" in associated_subject_level_deg_summary
                            else np.nan,
                            "subject_level_deg_status": ground_truth_subject_level_deg_summary.loc[0, "status"]
                            if not ground_truth_subject_level_deg_summary.empty and "status" in ground_truth_subject_level_deg_summary
                            else np.nan,
                            "pydeseq2_deg_comparison_status": pydeseq2_deg_comparison_summary.loc[0, "status"]
                            if not pydeseq2_deg_comparison_summary.empty and "status" in pydeseq2_deg_comparison_summary
                            else np.nan,
                            "supervised_status": supervised_summary.get("supervised_status") if supervised_summary else np.nan,
                            "n_ground_truth_positive": supervised_summary.get("n_ground_truth_positive") if supervised_summary else np.nan,
                            "n_ground_truth_negative": supervised_summary.get("n_ground_truth_negative") if supervised_summary else np.nan,
                            "mean_sim_ground_truth_positive": supervised_summary.get("mean_sim_ground_truth_positive") if supervised_summary else np.nan,
                            "mean_sim_ground_truth_negative": supervised_summary.get("mean_sim_ground_truth_negative") if supervised_summary else np.nan,
                            "ground_truth_mean_diff": supervised_summary.get("mean_diff") if supervised_summary else np.nan,
                            "ground_truth_mw_p": supervised_summary.get("mw_p") if supervised_summary else np.nan,
                            "ground_truth_roc_auc": supervised_summary.get("roc_auc") if supervised_summary else np.nan,
                            "pearson_gene_set": gene_summary.get("pearson"),
                            "spearman_gene_set": gene_summary.get("spearman"),
                            "mean_gene_set_score_associated": gene_summary.get("mean_gene_set_score_associated"),
                            "mean_gene_set_score_unassociated": gene_summary.get("mean_gene_set_score_unassociated"),
                            "mean_gene_set_score_difference": gene_summary.get("mean_gene_set_score_difference"),
                        }
                    )
                if supervised_rows_for_label:
                    supervised_df = pd.DataFrame(supervised_rows_for_label)
                    supervised_df.to_csv(label_dir / "supervised_celltype_score_summary.csv", index=False)
                    if config.plot and config.supervised_downstream_plots:
                        _plot_supervised_celltype_summary(
                            supervised_df,
                            label_dir / "distribution_difference_summary.pdf",
                        )

    summary = pd.DataFrame(summary_rows)
    summary_csv = run_dir / "downstream_summary.csv"
    summary.to_csv(summary_csv, index=False)
    rank_tail_metrics = (
        pd.concat(rank_tail_metric_frames, ignore_index=True)
        if rank_tail_metric_frames
        else pd.DataFrame()
    )
    rank_tail_model_summary = pd.DataFrame(rank_tail_summary_rows)
    if not rank_tail_metrics.empty:
        rank_tail_metrics.to_csv(run_dir / "rank_tail_enrichment_metrics.csv", index=False)
    if not rank_tail_model_summary.empty:
        rank_tail_model_summary.to_csv(run_dir / "rank_tail_enrichment_model_summary.csv", index=False)
    if not summary.empty and config.plot:
        fig, ax = plt.subplots(figsize=(max(3, 0.5 * summary["model_name"].nunique()), 2))
        sns.barplot(data=summary, x="model_name", y="n_associated", hue="functionality", ax=ax)
        sns.despine(ax=ax)
        fig.savefig(run_dir / "ablation_summary.pdf", bbox_inches="tight")
        plt.close(fig)
        _plot_model_gene_set_scatter_grid(
            gene_set_scatter_frames,
            run_dir / "model_score_vs_gene_set_score_by_model.pdf",
        )
        _plot_gene_set_correlation_barplot(
            summary,
            run_dir / "gene_set_spearman_ablation_summary.pdf",
            metric_column="spearman_gene_set",
            y_label="Spearman correlation",
        )
        _plot_gene_set_correlation_dotplot(
            summary,
            run_dir / "gene_set_spearman_ablation_summary_dot.pdf",
            metric_column="spearman_gene_set",
            y_label="Spearman correlation",
        )
        _plot_gene_set_correlation_barplot(
            summary,
            run_dir / "gene_set_pearson_ablation_summary.pdf",
            metric_column="pearson_gene_set",
            y_label="Pearson correlation",
        )
        _plot_gene_set_correlation_dotplot(
            summary,
            run_dir / "gene_set_pearson_ablation_summary_dot.pdf",
            metric_column="pearson_gene_set",
            y_label="Pearson correlation",
        )
    if config.plot and not rank_tail_metrics.empty:
        for functionality, functionality_metrics in rank_tail_metrics.groupby("functionality", sort=False):
            suffix = _safe_component(str(functionality))
            _plot_rank_tail_metric_line(
                functionality_metrics,
                run_dir / f"rank_tail_lift_ablation_line_{suffix}.pdf",
                metric_column="lift",
                y_label="Lift over random overlap",
                reference_line=1.0,
            )
            _plot_rank_tail_metric_line(
                functionality_metrics,
                run_dir / f"rank_tail_median_gene_set_percentile_ablation_line_{suffix}.pdf",
                metric_column="median_gene_set_percentile_rank",
                y_label="Median gene-set percentile rank",
                reference_line=0.5,
            )
    if config.plot and not rank_tail_model_summary.empty:
        for functionality, functionality_summary in rank_tail_model_summary.groupby("functionality", sort=False):
            suffix = _safe_component(str(functionality))
            _plot_rank_tail_metric_bar(
                functionality_summary,
                run_dir / f"rank_tail_lift_auc_ablation_bar_{suffix}.pdf",
                metric_column="area_under_lift_curve",
                y_label="Area under lift curve",
                reference_line=1.0,
            )
            _plot_rank_tail_metric_bar(
                functionality_summary,
                run_dir / f"rank_tail_median_gene_set_percentile_auc_ablation_bar_{suffix}.pdf",
                metric_column="area_under_median_gene_set_percentile_curve",
                y_label="Area under median percentile curve",
                reference_line=0.5,
            )
    write_metadata(
        run_dir,
        {
            "evaluation_name": "functionality_downstream_assessment",
            "config": _json_safe_config(config),
            "embedding_sources": {
                model: {
                    dataset: {
                        artifact: metadata["path"]
                        for artifact, metadata in dataset_meta.items()
                        if artifact in {"df_cells", "df_additional"}
                        and isinstance(metadata, dict)
                        and "path" in metadata
                    }
                    for dataset, dataset_meta in model_data.items()
                }
                for model, model_data in embeddings_dict.items()
            },
        },
    )
    return {"run_dir": run_dir, "summary_csv": summary_csv}
