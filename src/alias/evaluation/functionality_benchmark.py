from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from alias.util.plots.color_definition import (
    PUBLICATION_ABLATION_MODEL_LABELS,
    PUBLICATION_ABLATION_MODEL_ORDER,
    PUBLICATION_ABLATION_MODEL_PALETTE,
)


SourceFormat = Literal[
    "functionality_similarity",
    "score_matrix",
    "llm_matrix",
    "cellwhisperer_results",
]


@dataclass(frozen=True)
class FunctionalityBenchmarkSource:
    source_name: str
    path: Path | str
    source_format: SourceFormat
    assignment_level: str
    model_name: str
    score_column: str | None = None


@dataclass(frozen=True)
class FunctionalityBenchmarkConfig:
    output_dir: Path | str
    functionality_mapping_path: Path | str
    sources: list[FunctionalityBenchmarkSource]
    rank_ascending: bool = False
    accepted_label_columns: tuple[str, ...] = ("cell_type", "accepted_cell_type")
    timestamp: str | None = None


FUNCTIONALITY_COLUMNS = (
    "functionality",
    "Definition",
    "Functional Description",
    "functional_description",
)
CELL_TYPE_COLUMNS = (
    "cell_type",
    "Cell Type",
    "accepted_cell_type",
    "Accepted Cell Type",
)

ABLATION_ASSIGNMENT_LEVELS = {
    "cell": ("Cell-based assignment", "ablation_cell"),
    "celltype_label": ("Cell type label-based assignment", "ablation_celltype_label"),
}
ASSIGNMENT_GROUP_LABELS = {
    "cell": "Cell-based assignment",
    "cellwhisperer_cell": "Cell-based assignment",
    "celltype_label": "Cell type label-based assignment",
    "llm_label": "Cell type label-based assignment",
}
ALIAS_SOURCE_NAMES = {"ours_cell", "ours_label"}
ALIAS_MODEL_COLOR_ORDER = PUBLICATION_ABLATION_MODEL_ORDER
ALIAS_MODEL_PUBLICATION_LABELS = PUBLICATION_ABLATION_MODEL_LABELS
MB_BLUE = "#2171b5"
PLOT_HEIGHT = 1.75
BENCHMARK_PLOT_HEIGHT = 2.3
BENCHMARK_LEFT_MARGIN = 0.36
BENCHMARK_BOTTOM_MARGIN = 0.36
AXIS_LABEL_SIZE = 9.5
TICK_LABEL_SIZE = 8.5
BAR_X_SPACING = 0.46
BENCHMARK_BAR_X_SPACING = 0.55
BAR_WIDTH = 0.22
SWARM_X_SPACING = 0.54
PAIRED_BAR_X_SPACING = 0.62
PAIRED_BAR_WIDTH = 0.18
BENCHMARK_GROUP_MODEL_ORDER = {
    "Cell-based assignment": ["CellWhisperer", "MB"],
}
RANK_YTICKS = [1, 3, 5, 7, 9]


def _publication_model_label(model_name: str) -> str:
    return ALIAS_MODEL_PUBLICATION_LABELS.get(str(model_name), str(model_name))


def _publication_model_labels(model_names: list[str]) -> list[str]:
    return [_publication_model_label(model_name) for model_name in model_names]


def _timestamp_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")


def _safe_path_component(value: str) -> str:
    import re

    cleaned = re.sub(r"\s+", "_", str(value).strip())
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "", cleaned)
    return cleaned or "unknown"


def _unique_run_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir
    suffix = 1
    while True:
        candidate = base_dir.with_name(f"{base_dir.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate
        suffix += 1


def _first_present(columns: pd.Index | list[str], candidates: tuple[str, ...]) -> str | None:
    column_set = set(columns)
    return next((column for column in candidates if column in column_set), None)


def _coerce_source_path(path: Path | str) -> Path:
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Functionality benchmark source does not exist: {source_path}")
    return source_path


def _read_csv(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(_coerce_source_path(path))


def _standardize_scores(
    df: pd.DataFrame,
    *,
    source: FunctionalityBenchmarkSource,
    score_column: str,
) -> pd.DataFrame:
    functionality_column = _first_present(df.columns, FUNCTIONALITY_COLUMNS)
    cell_type_column = _first_present(df.columns, CELL_TYPE_COLUMNS)
    required = {
        "functionality": functionality_column,
        "cell_type": cell_type_column,
        "score": score_column if score_column in df.columns else None,
    }
    missing = [name for name, column in required.items() if column is None]
    if missing:
        raise ValueError(
            f"{source.path} is missing required {missing} columns for {source.source_format}."
        )

    out = pd.DataFrame(
        {
            "model_name": source.model_name,
            "source_name": source.source_name,
            "assignment_level": source.assignment_level,
            "functionality": df[functionality_column].astype(str),
            "cell_type": df[cell_type_column].astype(str),
            "score": pd.to_numeric(df[score_column], errors="coerce"),
        }
    )
    return out.dropna(subset=["score"]).reset_index(drop=True)


def _matrix_to_long(df: pd.DataFrame, source: FunctionalityBenchmarkSource) -> pd.DataFrame:
    functionality_column = _first_present(df.columns, FUNCTIONALITY_COLUMNS)
    cell_type_column = _first_present(df.columns, CELL_TYPE_COLUMNS)

    if cell_type_column is not None and functionality_column is None:
        long_df = df.melt(
            id_vars=cell_type_column,
            var_name="functionality",
            value_name="score",
        ).rename(columns={cell_type_column: "cell_type"})
    else:
        if functionality_column is None:
            functionality_column = df.columns[0]
        long_df = df.melt(
            id_vars=functionality_column,
            var_name="cell_type",
            value_name="score",
        ).rename(columns={functionality_column: "functionality"})

    return pd.DataFrame(
        {
            "model_name": source.model_name,
            "source_name": source.source_name,
            "assignment_level": source.assignment_level,
            "functionality": long_df["functionality"].astype(str),
            "cell_type": long_df["cell_type"].astype(str),
            "score": pd.to_numeric(long_df["score"], errors="coerce"),
        }
    ).dropna(subset=["score"]).reset_index(drop=True)


def load_source_scores(source: FunctionalityBenchmarkSource) -> pd.DataFrame:
    """Load one benchmark source into the canonical long score table."""
    df = _read_csv(source.path)
    if source.source_format == "functionality_similarity":
        score_column = source.score_column or "mean_auc"
        return _standardize_scores(df, source=source, score_column=score_column)
    if source.source_format == "cellwhisperer_results":
        score_column = source.score_column or "mean_auc"
        return _standardize_scores(df, source=source, score_column=score_column)
    if source.source_format in {"score_matrix", "llm_matrix"}:
        return _matrix_to_long(df, source)
    raise ValueError(f"Unknown functionality benchmark source format: {source.source_format}")


def load_functionality_mapping(
    path: Path | str,
    *,
    accepted_label_columns: tuple[str, ...] = ("cell_type", "accepted_cell_type"),
) -> pd.DataFrame:
    """Load functionality-to-celltype ground truth and normalize column names."""
    df = _read_csv(path)
    functionality_column = _first_present(df.columns, FUNCTIONALITY_COLUMNS)
    if functionality_column is None:
        raise ValueError(f"{path} is missing a functionality/Definition column.")

    accepted_columns = [
        column
        for column in accepted_label_columns
        if column in df.columns
    ]
    accepted_columns.extend(
        column
        for column in CELL_TYPE_COLUMNS
        if column in df.columns and column not in accepted_columns
    )
    if not accepted_columns:
        raise ValueError(f"{path} is missing an accepted cell type column.")

    rows: list[dict[str, str]] = []
    for _, row in df.iterrows():
        functionality = str(row[functionality_column])
        for column in accepted_columns:
            value = row[column]
            if pd.isna(value):
                continue
            for label in str(value).split(";"):
                clean_label = label.strip()
                if clean_label:
                    rows.append(
                        {
                            "functionality": functionality,
                            "accepted_cell_type": clean_label,
                        }
                    )

    mapping = pd.DataFrame(rows).drop_duplicates()
    if mapping.empty:
        raise ValueError(f"{path} does not contain any functionality mappings.")
    return mapping.reset_index(drop=True)


def _rank_scores(scores: pd.DataFrame, *, rank_ascending: bool) -> pd.DataFrame:
    ranked = scores.copy()
    ranked["rank"] = (
        ranked.groupby(["model_name", "source_name", "assignment_level", "functionality"])["score"]
        .rank(method="min", ascending=rank_ascending)
        .astype(float)
    )
    ranked["n_ranked_cell_types"] = ranked.groupby(
        ["model_name", "source_name", "assignment_level", "functionality"]
    )["cell_type"].transform("nunique")
    return ranked


def compute_ground_truth_ranks(
    scores: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    rank_ascending: bool = False,
) -> pd.DataFrame:
    """Extract the rank of the accepted cell type(s) for each functionality."""
    required_score_columns = {
        "model_name",
        "source_name",
        "assignment_level",
        "functionality",
        "cell_type",
        "score",
    }
    missing_scores = required_score_columns.difference(scores.columns)
    if missing_scores:
        raise ValueError(f"Scores are missing required columns: {sorted(missing_scores)}")

    mapping = mapping.copy()
    if "accepted_cell_type" not in mapping.columns and "cell_type" in mapping.columns:
        mapping = mapping.rename(columns={"cell_type": "accepted_cell_type"})

    if not {"functionality", "accepted_cell_type"}.issubset(mapping.columns):
        raise ValueError("Mapping must contain `functionality` and `accepted_cell_type` columns.")

    ranked = _rank_scores(scores, rank_ascending=rank_ascending)
    source_keys = ranked[["model_name", "source_name", "assignment_level"]].drop_duplicates()
    functionality_keys = mapping[["functionality"]].drop_duplicates()
    expected = source_keys.merge(functionality_keys, how="cross").merge(mapping, on="functionality", how="left")

    merged = expected.merge(
        ranked,
        left_on=[
            "model_name",
            "source_name",
            "assignment_level",
            "functionality",
            "accepted_cell_type",
        ],
        right_on=[
            "model_name",
            "source_name",
            "assignment_level",
            "functionality",
            "cell_type",
        ],
        how="left",
    )

    grouped_rows: list[dict[str, object]] = []
    group_columns = ["model_name", "source_name", "assignment_level", "functionality"]
    for group_key, group in merged.groupby(group_columns, sort=False, dropna=False):
        present = group.dropna(subset=["rank"])
        accepted_labels = sorted(group["accepted_cell_type"].dropna().astype(str).unique())
        if present.empty:
            ranked_group = ranked[
                (ranked["model_name"] == group_key[0])
                & (ranked["source_name"] == group_key[1])
                & (ranked["assignment_level"] == group_key[2])
                & (ranked["functionality"] == group_key[3])
            ]
            n_ranked = int(ranked_group["cell_type"].nunique()) if not ranked_group.empty else 0
            grouped_rows.append(
                {
                    "model_name": group_key[0],
                    "source_name": group_key[1],
                    "assignment_level": group_key[2],
                    "functionality": group_key[3],
                    "accepted_cell_types": ";".join(accepted_labels),
                    "best_cell_type": np.nan,
                    "score": np.nan,
                    "rank": np.nan,
                    "n_ranked_cell_types": n_ranked,
                    "normalized_rank": np.nan,
                    "reciprocal_rank": np.nan,
                    "status": "missing_score",
                }
            )
            continue

        best = present.sort_values(["rank", "score"], ascending=[True, rank_ascending]).iloc[0]
        n_ranked = int(best["n_ranked_cell_types"])
        rank = float(best["rank"])
        normalized_rank = rank / n_ranked if n_ranked else np.nan
        grouped_rows.append(
            {
                "model_name": group_key[0],
                "source_name": group_key[1],
                "assignment_level": group_key[2],
                "functionality": group_key[3],
                "accepted_cell_types": ";".join(accepted_labels),
                "best_cell_type": str(best["accepted_cell_type"]),
                "score": float(best["score"]),
                "rank": rank,
                "n_ranked_cell_types": n_ranked,
                "normalized_rank": normalized_rank,
                "reciprocal_rank": 1.0 / rank,
                "status": "ok",
            }
        )

    return pd.DataFrame(grouped_rows)


def summarize_ground_truth_ranks(ranks: pd.DataFrame) -> pd.DataFrame:
    if ranks.empty:
        return pd.DataFrame()

    def _top_k(series: pd.Series, k: int) -> float:
        if series.empty:
            return np.nan
        return float(series.le(k).fillna(False).mean())

    summary = (
        ranks.groupby(["model_name", "source_name", "assignment_level"], dropna=False)
        .agg(
            n_functionalities=("functionality", "nunique"),
            n_ok=("status", lambda values: int((values == "ok").sum())),
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            mean_normalized_rank=("normalized_rank", "mean"),
            mrr=("reciprocal_rank", "mean"),
            top1_accuracy=("rank", lambda values: _top_k(values, 1)),
            top3_accuracy=("rank", lambda values: _top_k(values, 3)),
            top5_accuracy=("rank", lambda values: _top_k(values, 5)),
        )
        .reset_index()
    )
    summary["missing_score_count"] = summary["n_functionalities"] - summary["n_ok"]
    return summary


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("pdf", "png"):
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(output_dir / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def _model_order(values: pd.Series) -> list[str]:
    present = [str(value) for value in values.dropna().astype(str).unique()]
    ordered = [model for model in ALIAS_MODEL_COLOR_ORDER if model in set(present)]
    ordered.extend(model for model in present if model not in set(ordered))
    return ordered


def _benchmark_model_palette(model_names: list[str]) -> dict[str, str | tuple[float, float, float, float]]:
    palette: dict[str, str | tuple[float, float, float, float]] = {}
    purple_models = [model for model in model_names if model != "MB"]
    purples = sns.color_palette("Purples", n_colors=max(3, len(purple_models) + 2))[2:]
    purple_lookup = {
        model: purples[index]
        for index, model in enumerate(purple_models)
    }
    for model in model_names:
        palette[model] = MB_BLUE if model == "MB" else purple_lookup[model]
    return palette


def _ablation_model_palette(model_names: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("Blues")
    reference_order = ALIAS_MODEL_COLOR_ORDER
    full_palette = {
        model: PUBLICATION_ABLATION_MODEL_PALETTE[model]
        for model in reference_order
    }
    extra_models = [model for model in model_names if model not in full_palette]
    if extra_models:
        fallback_values = [
            0.35 + index * (0.45 / max(1, len(extra_models) - 1))
            for index in range(len(extra_models))
        ]
        full_palette.update(
            {model: cmap(value) for model, value in zip(extra_models, fallback_values)}
        )
    return {model: full_palette[model] for model in model_names}


def _add_assignment_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["assignment_group"] = out["assignment_level"].map(ASSIGNMENT_GROUP_LABELS)
    return out.dropna(subset=["assignment_group"])


def _filter_benchmark_comparison_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Keep MB as the only ALIAS model in cross-method benchmark comparison plots."""
    is_alias_source = summary["source_name"].isin(ALIAS_SOURCE_NAMES)
    keep = ~is_alias_source | summary["model_name"].astype(str).eq("MB")
    return summary.loc[keep].copy()


def _even_offsets(n_values: int, *, width: float = 0.18) -> np.ndarray:
    if n_values <= 1:
        return np.array([0.0])
    return np.linspace(-width, width, n_values)


def _plot_rank_boxstrip(
    plot_df: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    x_order: list[str],
    palette_by_key: dict[str, str | tuple[float, float, float, float]],
    ax: plt.Axes,
    box_width: float = 0.28,
    jitter_width: float = 0.075,
    point_size: float = 9,
) -> None:
    grouped_values = [
        plot_df.loc[plot_df[x_column].astype(str) == key, y_column].dropna().astype(float).to_numpy()
        for key in x_order
    ]
    positions = np.arange(len(x_order), dtype=float)
    nonempty_positions = [
        position
        for position, values in zip(positions, grouped_values)
        if len(values) > 0
    ]
    nonempty_values = [values for values in grouped_values if len(values) > 0]
    nonempty_keys = [key for key, values in zip(x_order, grouped_values) if len(values) > 0]
    if not nonempty_values:
        return

    boxplot = ax.boxplot(
        nonempty_values,
        positions=nonempty_positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "0.15", "linewidth": 0.8},
        whiskerprops={"color": "0.25", "linewidth": 0.7},
        capprops={"color": "0.25", "linewidth": 0.7},
        boxprops={"edgecolor": "0.25", "linewidth": 0.7},
    )
    for patch, key in zip(boxplot["boxes"], nonempty_keys):
        patch.set_facecolor(palette_by_key[key])
        patch.set_alpha(0.28)

    rng = np.random.default_rng(42)
    for position, key, values in zip(nonempty_positions, nonempty_keys, nonempty_values):
        jitter = rng.uniform(-jitter_width, jitter_width, size=len(values))
        ax.scatter(
            position + jitter,
            values,
            color=palette_by_key[key],
            edgecolors="0.15",
            linewidths=0.2,
            alpha=0.82,
            s=point_size,
            zorder=3,
        )


def _plot_even_rank_dots(
    plot_df: pd.DataFrame,
    *,
    x_order: list[str],
    palette_by_key: dict[str, str | tuple[float, float, float, float]],
    ax: plt.Axes,
) -> None:
    for x_index, plot_key in enumerate(x_order):
        values = (
            plot_df.loc[plot_df["plot_key"].astype(str) == plot_key, "rank"]
            .dropna()
            .astype(float)
            .sort_values(kind="mergesort")
            .to_numpy()
        )
        if len(values) == 0:
            continue
        offsets = _even_offsets(len(values))
        ax.scatter(
            x_index + offsets,
            values,
            color=palette_by_key[plot_key],
            edgecolor="0.15",
            linewidth=0.2,
            s=10,
            alpha=0.9,
        )


def _benchmark_assignment_x_order(plot_df: pd.DataFrame) -> list[str]:
    assignment_order = [
        "Cell-based assignment",
        "Cell type label-based assignment",
    ]
    default_model_order = _model_order(plot_df["model_name"])
    x_order: list[str] = []
    for assignment_group in assignment_order:
        present_models = [
            model
            for model in default_model_order
            if (
                (plot_df["assignment_group"] == assignment_group)
                & (plot_df["model_name"].astype(str) == model)
            ).any()
        ]
        preferred = BENCHMARK_GROUP_MODEL_ORDER.get(assignment_group, [])
        ordered_models = [
            model for model in preferred if model in set(present_models)
        ]
        ordered_models.extend(model for model in present_models if model not in set(ordered_models))
        x_order.extend(f"{assignment_group}__{model}" for model in ordered_models)
    return x_order


def _plot_rank_distribution(ranks: pd.DataFrame, output_dir: Path) -> None:
    plot_df = ranks[ranks["status"] == "ok"].copy()
    if plot_df.empty:
        return
    plot_df["method"] = plot_df["model_name"].astype(str) + " / " + plot_df["source_name"].astype(str)
    method_order = plot_df["method"].drop_duplicates().astype(str).tolist()
    palette = {method: "#2171b5" for method in method_order}
    fig, ax = plt.subplots(figsize=(max(3.0, 0.45 * plot_df["method"].nunique()), PLOT_HEIGHT))
    _plot_rank_boxstrip(
        plot_df,
        x_column="method",
        y_column="rank",
        x_order=method_order,
        palette_by_key=palette,
        ax=ax,
        point_size=7,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Rank of correct label", fontsize=AXIS_LABEL_SIZE)
    ax.set_yticks(RANK_YTICKS)
    ax.set_xticks(range(len(method_order)))
    ax.set_xticklabels(method_order, rotation=45, ha="right", fontsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="x", which="major", bottom=True, length=3, width=0.7, color="0.2")
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    sns.despine(ax=ax)
    _save_figure(fig, output_dir, "benchmark_rank_distribution")


def _plot_ablation_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    for assignment_level, (assignment_label, stem_prefix) in ABLATION_ASSIGNMENT_LEVELS.items():
        plot_df = summary[
            (summary["source_name"].isin(["ours_cell", "ours_label"]))
            & (summary["assignment_level"] == assignment_level)
        ].copy()
        if plot_df.empty:
            continue

        model_order = _model_order(plot_df["model_name"])
        palette = _ablation_model_palette(model_order)
        plot_df["model_name"] = pd.Categorical(
            plot_df["model_name"],
            categories=model_order,
            ordered=True,
        )
        plot_df = plot_df.sort_values("model_name")

        x_positions = [index * BAR_X_SPACING for index in range(len(model_order))]
        values = [
            float(plot_df.loc[plot_df["model_name"].astype(str) == model, "mrr"].iloc[0])
            for model in model_order
        ]

        fig, ax = plt.subplots(figsize=(max(2.25, 0.46 * len(model_order) + 0.6), PLOT_HEIGHT))
        ax.bar(
            x_positions,
            values,
            width=BAR_WIDTH,
            color=[palette[model] for model in model_order],
            edgecolor="0.15",
            linewidth=0.35,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Mean reciprocal rank", fontsize=AXIS_LABEL_SIZE)
        ax.set_title("")
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            _publication_model_labels(model_order),
            rotation=45,
            ha="right",
            fontsize=TICK_LABEL_SIZE,
        )
        ax.tick_params(axis="x", which="major", bottom=True, length=3, width=0.7, color="0.2")
        ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
        if x_positions:
            ax.set_xlim(min(x_positions) - 0.24, max(x_positions) + 0.24)
        sns.despine(ax=ax)
        _save_figure(fig, output_dir, f"{stem_prefix}_mrr_bar")


def _plot_ablation_joint_assignment_bar(summary: pd.DataFrame, output_dir: Path) -> None:
    plot_df = summary[
        summary["source_name"].isin(["ours_cell", "ours_label"])
        & summary["assignment_level"].isin(["cell", "celltype_label"])
    ].copy()
    if plot_df.empty:
        return

    wide = plot_df.pivot_table(
        index="model_name",
        columns="assignment_level",
        values="mrr",
        aggfunc="first",
    ).dropna(subset=["cell", "celltype_label"], how="any")
    if wide.empty:
        return

    model_order = [model for model in _model_order(pd.Series(wide.index.astype(str))) if model in wide.index]
    wide = wide.loc[model_order]
    palette = _ablation_model_palette(model_order)
    x_positions = np.arange(len(model_order)) * PAIRED_BAR_X_SPACING

    fig, ax = plt.subplots(figsize=(3.25, PLOT_HEIGHT))
    ax.bar(
        x_positions - PAIRED_BAR_WIDTH / 2,
        wide["cell"].astype(float),
        width=PAIRED_BAR_WIDTH,
        color=[palette[model] for model in model_order],
        edgecolor="0.15",
        linewidth=0.35,
        label="Cell-based assignment",
    )
    ax.bar(
        x_positions + PAIRED_BAR_WIDTH / 2,
        wide["celltype_label"].astype(float),
        width=PAIRED_BAR_WIDTH,
        color=[palette[model] for model in model_order],
        edgecolor="0.15",
        linewidth=0.35,
        hatch="////",
        label="Cell type label-based assignment",
    )
    for patch in ax.patches[len(model_order):]:
        patch._hatch_color = (1, 1, 1, 1)

    ax.set_xlabel("")
    ax.set_ylabel("Mean reciprocal rank", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
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
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    sns.despine(ax=ax)
    _save_figure(fig, output_dir, "ablation_cell_vs_celltype_label_mrr_bar")


def _plot_ablation_rank_swarms(ranks: pd.DataFrame, output_dir: Path) -> None:
    if ranks.empty:
        return
    for assignment_level, (assignment_label, stem_prefix) in ABLATION_ASSIGNMENT_LEVELS.items():
        plot_df = ranks[
            (ranks["status"] == "ok")
            & (ranks["source_name"].isin(["ours_cell", "ours_label"]))
            & (ranks["assignment_level"] == assignment_level)
        ].copy()
        if plot_df.empty:
            continue

        model_order = _model_order(plot_df["model_name"])
        palette = _ablation_model_palette(model_order)
        plot_df["model_name"] = pd.Categorical(
            plot_df["model_name"],
            categories=model_order,
            ordered=True,
        )
        plot_df = plot_df.sort_values("model_name")

        fig, ax = plt.subplots(figsize=(max(3.0, 0.54 * len(model_order) + 0.6), PLOT_HEIGHT))
        _plot_rank_boxstrip(
            plot_df.assign(model_name=plot_df["model_name"].astype(str)),
            x_column="model_name",
            y_column="rank",
            x_order=model_order,
            palette_by_key=palette,
            ax=ax,
            point_size=8,
        )
        ax.set_xlabel("")
        ax.set_ylabel("Rank of correct label", fontsize=AXIS_LABEL_SIZE)
        ax.set_title("")
        ax.set_yticks(RANK_YTICKS)
        ax.set_xticks(range(len(model_order)))
        ax.set_xticklabels(
            _publication_model_labels(model_order),
            rotation=45,
            ha="right",
            fontsize=TICK_LABEL_SIZE,
        )
        ax.tick_params(axis="x", which="major", bottom=True, length=3, width=0.7, color="0.2")
        ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
        ax.invert_yaxis()
        sns.despine(ax=ax)
        _save_figure(fig, output_dir, f"{stem_prefix}_rank_swarm")


def _plot_benchmark_assignment_group_bar(summary: pd.DataFrame, output_dir: Path) -> None:
    plot_df = _add_assignment_group(_filter_benchmark_comparison_summary(summary))
    if plot_df.empty:
        return

    plot_df["plot_key"] = (
        plot_df["assignment_group"].astype(str)
        + "__"
        + plot_df["model_name"].astype(str)
    )
    model_order = _model_order(plot_df["model_name"])
    x_order = _benchmark_assignment_x_order(plot_df)
    palette_by_key = {
        f"{assignment_group}__{model}": color
        for model, color in _benchmark_model_palette(model_order).items()
        for assignment_group in ASSIGNMENT_GROUP_LABELS.values()
    }
    x_labels = [_publication_model_label(key.split("__", maxsplit=1)[1]) for key in x_order]

    plot_df["plot_key"] = pd.Categorical(plot_df["plot_key"], categories=x_order, ordered=True)
    plot_df = plot_df.sort_values("plot_key", kind="mergesort")
    x_positions = [index * BENCHMARK_BAR_X_SPACING for index in range(len(x_order))]
    values = [
        float(plot_df.loc[plot_df["plot_key"].astype(str) == key, "mrr"].iloc[0])
        for key in x_order
    ]

    fig_width = max(2.25, BENCHMARK_BAR_X_SPACING * len(x_order) + 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, BENCHMARK_PLOT_HEIGHT))
    ax.bar(
        x_positions,
        values,
        width=BAR_WIDTH,
        color=[palette_by_key[key] for key in x_order],
        edgecolor="0.15",
        linewidth=0.35,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean reciprocal rank", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="x", which="major", bottom=True, length=3, width=0.7, color="0.2")
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    if x_positions:
        ax.set_xlim(min(x_positions) - 0.24, max(x_positions) + 0.24)
    fig.subplots_adjust(left=BENCHMARK_LEFT_MARGIN, bottom=BENCHMARK_BOTTOM_MARGIN)
    sns.despine(ax=ax)
    _save_figure(fig, output_dir, "benchmark_assignment_group_mrr_bar")


def _plot_benchmark_assignment_group_rank_plot(ranks: pd.DataFrame, output_dir: Path) -> None:
    plot_df = ranks[ranks["status"] == "ok"].copy()
    plot_df = _add_assignment_group(_filter_benchmark_comparison_summary(plot_df))
    if plot_df.empty:
        return

    plot_df["plot_key"] = (
        plot_df["assignment_group"].astype(str)
        + "__"
        + plot_df["model_name"].astype(str)
    )
    model_order = _model_order(plot_df["model_name"])
    x_order = _benchmark_assignment_x_order(plot_df)
    palette_by_key = {
        f"{assignment_group}__{model}": color
        for model, color in _benchmark_model_palette(model_order).items()
        for assignment_group in ASSIGNMENT_GROUP_LABELS.values()
    }
    x_labels = [_publication_model_label(key.split("__", maxsplit=1)[1]) for key in x_order]

    plot_df["plot_key"] = pd.Categorical(plot_df["plot_key"], categories=x_order, ordered=True)
    plot_df = plot_df.sort_values("plot_key", kind="mergesort").assign(
        plot_key=lambda frame: frame["plot_key"].astype(str)
    )
    fig_width = max(2.25, BENCHMARK_BAR_X_SPACING * len(x_order) + 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, BENCHMARK_PLOT_HEIGHT))
    _plot_even_rank_dots(
        plot_df,
        x_order=x_order,
        palette_by_key=palette_by_key,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Rank of correct label", fontsize=AXIS_LABEL_SIZE)
    ax.set_yticks(RANK_YTICKS)
    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="x", which="major", bottom=True, length=3, width=0.7, color="0.2")
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    ax.invert_yaxis()
    fig.subplots_adjust(left=BENCHMARK_LEFT_MARGIN, bottom=BENCHMARK_BOTTOM_MARGIN)
    sns.despine(ax=ax)
    _save_figure(fig, output_dir, "benchmark_assignment_group_rank_plot")
    fig, ax = plt.subplots(figsize=(fig_width, BENCHMARK_PLOT_HEIGHT))
    _plot_rank_boxstrip(
        plot_df,
        x_column="plot_key",
        y_column="rank",
        x_order=x_order,
        palette_by_key=palette_by_key,
        ax=ax,
        point_size=8,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Rank of correct label", fontsize=AXIS_LABEL_SIZE)
    ax.set_yticks(RANK_YTICKS)
    ax.set_xticks(range(len(x_order)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="x", which="major", bottom=True, length=3, width=0.7, color="0.2")
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    ax.invert_yaxis()
    fig.subplots_adjust(left=BENCHMARK_LEFT_MARGIN, bottom=BENCHMARK_BOTTOM_MARGIN)
    sns.despine(ax=ax)
    _save_figure(fig, output_dir, "benchmark_assignment_group_rank_swarm")


def _plot_rank_heatmap(ranks: pd.DataFrame, output_dir: Path) -> None:
    plot_df = ranks[ranks["status"] == "ok"].copy()
    if plot_df.empty:
        return
    plot_df["method"] = plot_df["model_name"].astype(str) + " / " + plot_df["source_name"].astype(str)
    heatmap_df = plot_df.pivot_table(
        index="method",
        columns="functionality",
        values="rank",
        aggfunc="min",
    )
    fig_width = max(6.0, 0.35 * len(heatmap_df.columns) + 2.0)
    fig_height = max(2.5, 0.35 * len(heatmap_df.index) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        heatmap_df,
        cmap="viridis_r",
        annot=True,
        fmt=".0f",
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Rank"},
        ax=ax,
    )
    ax.set_xlabel("Functionality", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=45, labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    _save_figure(fig, output_dir, "benchmark_true_label_rank_heatmap")


def _make_run_dir(output_dir: Path | str, timestamp: str | None) -> Path:
    run_dir = _unique_run_dir(Path(output_dir) / _safe_path_component(timestamp or _timestamp_now()))
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _json_safe_config(config: FunctionalityBenchmarkConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["output_dir"] = str(config.output_dir)
    payload["functionality_mapping_path"] = str(config.functionality_mapping_path)
    payload["sources"] = [
        {**asdict(source), "path": str(source.path)}
        for source in config.sources
    ]
    return payload


def run_functionality_benchmark(config: FunctionalityBenchmarkConfig) -> dict[str, Path]:
    """Run rank-based functionality benchmarking and write evaluation artifacts."""
    run_dir = _make_run_dir(config.output_dir, config.timestamp)
    mapping = load_functionality_mapping(
        config.functionality_mapping_path,
        accepted_label_columns=config.accepted_label_columns,
    )
    source_tables = [load_source_scores(source) for source in config.sources]
    all_scores = pd.concat(source_tables, ignore_index=True) if source_tables else pd.DataFrame()
    if all_scores.empty:
        raise ValueError("No functionality benchmark scores were loaded.")

    ranks = compute_ground_truth_ranks(
        all_scores,
        mapping,
        rank_ascending=config.rank_ascending,
    )
    summary = summarize_ground_truth_ranks(ranks)

    all_scores_csv = run_dir / "all_scores_long.csv"
    ranks_csv = run_dir / "ground_truth_ranks.csv"
    summary_csv = run_dir / "benchmark_summary.csv"
    metadata_json = run_dir / "benchmark_metadata.json"

    all_scores.to_csv(all_scores_csv, index=False)
    ranks.to_csv(ranks_csv, index=False)
    summary.to_csv(summary_csv, index=False)

    _plot_rank_distribution(ranks, run_dir)
    _plot_ablation_rank_swarms(ranks, run_dir)
    _plot_ablation_summary(summary, run_dir)
    _plot_ablation_joint_assignment_bar(summary, run_dir)
    _plot_benchmark_assignment_group_bar(summary, run_dir)
    _plot_benchmark_assignment_group_rank_plot(ranks, run_dir)
    _plot_rank_heatmap(ranks, run_dir)

    metadata = {
        "evaluation_name": "functionality_benchmark",
        "run_dir": str(run_dir),
        "mapping_path": str(config.functionality_mapping_path),
        "n_sources": len(config.sources),
        "n_scores": int(len(all_scores)),
        "n_ground_truth_rows": int(len(ranks)),
        "config": _json_safe_config(config),
        "outputs": {
            "all_scores_long": str(all_scores_csv),
            "ground_truth_ranks": str(ranks_csv),
            "benchmark_summary": str(summary_csv),
        },
    }
    with metadata_json.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    return {
        "run_dir": run_dir,
        "all_scores_csv": all_scores_csv,
        "ground_truth_ranks_csv": ranks_csv,
        "summary_csv": summary_csv,
        "metadata_json": metadata_json,
    }
