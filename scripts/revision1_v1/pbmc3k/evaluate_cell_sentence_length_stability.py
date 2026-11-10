from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

HIAI_TCELLS_SCRIPT_DIR = PROJECT_ROOT / "scripts" / "revision1_v1" / "HIAI_Tcells"
if str(HIAI_TCELLS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(HIAI_TCELLS_SCRIPT_DIR))

import anndata as ad
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import Dataset, concatenate_datasets, load_from_disk
import pyarrow as pa
import pyarrow.ipc as pa_ipc

import alias.evaluation as ev
from alias.evaluation.embedding_evaluation.external import align_embedding_frame_to_adata
from alias.util.plots.color_definition import (
    CATEGORICAL_PALETTES,
    PBMC3K_CELLTYPE_POINT_COLOR,
    PBMC3K_SCATTER_EDGE_COLOR,
)
from util.publication_plotting import (
    AXIS_LABEL_SIZE,
    BAR_EDGE_COLOR,
    BAR_EDGE_LINEWIDTH,
    BAR_WIDTH,
    PLOT_HEIGHT,
    TICK_LABEL_SIZE,
    save_publication_figure,
    set_publication_style,
)


STUDY_ID = "pbmc3k_cell_sentence_length_stability"
DATASET_ID = STUDY_ID
ANNOTATION_COLUMN = "louvain"
LABEL_KEY = "cell_type"
BATCH_KEY = "pbmc3k_single_batch"
SCIB_SCORE_COLUMNS = ["Bio conservation"]
CS_LENGTHS = [10, 20, 50, 100, 200]
TRAINING_SEEDS = [73, 101, 202]
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_CELLS = 20_000
N_JOBS = int(os.environ.get("PBMC3K_SCIB_N_JOBS", "1"))

STUDY_ROOT = Path(
    os.environ.get(
        "PBMC3K_STUDY_ROOT", PROJECT_ROOT / "out" / "data" / "revision1_v1" / "pbmc3k"
    )
)
MODEL_ROOT = Path(os.environ.get("PBMC3K_MODEL_ROOT", PROJECT_ROOT / "models"))
DATA_ROOT = STUDY_ROOT / "datasets"
OUTPUT_ROOT = Path(
    os.environ.get(
        "PBMC3K_EVALUATION_RUN_DIR",
        STUDY_ROOT
        / "analysis_runs"
        / "cell_sentence_length_stability"
        / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{os.getpid()}",
    )
)
EMBEDDINGS_ROOT = OUTPUT_ROOT / "embeddings"
LABEL_ALIGNMENT_ROOT = OUTPUT_ROOT / "label_alignment"
SCIB_ROOT = OUTPUT_ROOT / "scib_bioconservation"
PLOTS_ROOT = OUTPUT_ROOT / "plots"


@dataclass(frozen=True)
class ModelSpec:
    cs_length: int
    seed: int
    model_path: Path | None = None

    @property
    def dataset_id(self) -> str:
        return f"pbmc3k_cs{self.cs_length}"

    @property
    def model_id(self) -> str:
        return f"{self.dataset_id}_seed{self.seed}"

    @property
    def local_model_path(self) -> Path:
        return self.model_path or MODEL_ROOT / self.model_id

    @property
    def model_reference(self) -> str:
        return str(self.local_model_path)

    @property
    def scrna_test_path(self) -> Path:
        return DATA_ROOT / self.dataset_id / "datasets" / "scrna_test"

    @property
    def adata_test_path(self) -> Path:
        return DATA_ROOT / self.dataset_id / "adata_test.h5ad"


def build_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(cs_length=cs_length, seed=seed)
        for cs_length in CS_LENGTHS
        for seed in TRAINING_SEEDS
    ]


def select_model_specs(
    specs: list[ModelSpec], *, cs_length: int | str | None, training_seed: int | str | None
) -> list[ModelSpec]:
    def selected_values(value: int | str | None, allowed: list[int], option: str) -> set[int]:
        if value is None or value == "all":
            return set(allowed)
        tokens = str(value).split(",")
        if any(not token.isdecimal() for token in tokens):
            raise ValueError(f"Invalid {option}: {value}. Use 'all' or comma-separated values from {allowed}.")
        selected = {int(token) for token in tokens}
        if not selected.issubset(allowed):
            raise ValueError(f"Invalid {option}: {value}. Choose from {allowed}.")
        return selected

    lengths = selected_values(cs_length, CS_LENGTHS, "--cs-length")
    seeds = selected_values(training_seed, TRAINING_SEEDS, "--training-seed")
    selected = [
        spec
        for spec in specs
        if spec.cs_length in lengths and spec.seed in seeds
    ]
    if not selected:
        raise ValueError("No PBMC3k models matched the selected length and seed.")
    return selected


def require_local_models(specs: list[ModelSpec]) -> None:
    missing = [
        str(spec.local_model_path)
        for spec in specs
        if not (spec.local_model_path / "model.safetensors").is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing local PBMC3k model weights:\n" + "\n".join(missing))


def cs_length_palette(cs_lengths: list[int]) -> dict[int, str]:
    colors = CATEGORICAL_PALETTES["pbmc3k_cs_length_ropu"]
    ordered = [length for length in CS_LENGTHS if length in set(cs_lengths)]
    if len(ordered) > len(colors):
        raise ValueError("Not enough configured PBMC3k cs_length colors.")
    return {
        length: colors[idx]
        for idx, length in enumerate(ordered)
    }


def _load_arrow_dataset(path: Path) -> Dataset:
    arrow_files = sorted(path.glob("*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No Arrow data files found in {path}")

    datasets = []
    for arrow_file in arrow_files:
        with pa.memory_map(str(arrow_file), "r") as source:
            table = pa_ipc.open_stream(source).read_all()
        datasets.append(Dataset.from_pandas(table.to_pandas(), preserve_index=False))
    return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)


def _load_scrna_test(spec: ModelSpec) -> Dataset:
    if not spec.scrna_test_path.exists():
        raise FileNotFoundError(f"Missing scRNA test dataset: {spec.scrna_test_path}")
    try:
        return load_from_disk(str(spec.scrna_test_path))
    except Exception as error:
        print(
            f"Could not load local scrna_test from {spec.scrna_test_path}: "
            f"{type(error).__name__}: {error}"
        )
        print("Reading local scrna_test Arrow files directly instead.")
        try:
            return _load_arrow_dataset(spec.scrna_test_path)
        except Exception as arrow_error:
            raise RuntimeError(
                f"Local scrna_test exists but could not be loaded from {spec.scrna_test_path}. "
                f"Current working directory: {Path.cwd()}. "
                f"load_from_disk error: {type(error).__name__}: {error}. "
                f"Arrow fallback error: {type(arrow_error).__name__}: {arrow_error}"
            ) from arrow_error


def _load_adata_test(spec: ModelSpec) -> ad.AnnData:
    if not spec.adata_test_path.exists():
        raise FileNotFoundError(f"Missing AnnData test artifact: {spec.adata_test_path}")
    adata = ad.read_h5ad(spec.adata_test_path)
    adata.obs_names = adata.obs_names.astype(str)
    if ANNOTATION_COLUMN not in adata.obs:
        raise KeyError(f"Missing annotation column in adata.obs: {ANNOTATION_COLUMN}")
    adata.obs[LABEL_KEY] = adata.obs[ANNOTATION_COLUMN].astype(str).astype("category")
    adata.obs[BATCH_KEY] = "pbmc3k"
    adata.obs[BATCH_KEY] = adata.obs[BATCH_KEY].astype("category")
    return adata


def _embedding_config(spec: ModelSpec, *, force: bool, batch_size: int, max_cells: int) -> ev.GenEmbeddingsConfig:
    return ev.GenEmbeddingsConfig(
        embedding_models=[spec.model_reference],
        model_output_names={spec.model_reference: spec.model_id},
        model_type="sentence_transformer",
        batch_size=batch_size,
        output_dir=str(EMBEDDINGS_ROOT / spec.dataset_id),
        max_cells=max_cells,
        annotation_column=ANNOTATION_COLUMN,
        force_regenerate=force,
    )


def generate_model_embeddings(
    spec: ModelSpec,
    *,
    force: bool,
    batch_size: int,
    max_cells: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    scrna_test = _load_scrna_test(spec)
    evaluation_dict = {"scrna": {"test": scrna_test}}
    return ev.generate_embeddings(
        evaluation_dict=evaluation_dict,
        embedding_config=_embedding_config(
            spec,
            force=force,
            batch_size=batch_size,
            max_cells=max_cells,
        ),
    )


def run_label_alignment(
    spec: ModelSpec,
    embeddings_dict: dict[str, dict[str, dict[str, Any]]],
) -> pd.DataFrame:
    output_dir = LABEL_ALIGNMENT_ROOT / spec.dataset_id / spec.model_id
    similarity_config = ev.CellTypeSimilarityConfig(
        similarity_metric="cosine",
        bins=60,
        output_dir=output_dir,
    )
    similarity_df = ev.cell_type_label_similarity(
        embeddings_dict=embeddings_dict,
        annotation_column=ANNOTATION_COLUMN,
        config=similarity_config,
    )
    similarity_df = similarity_df.copy()
    similarity_df["model_id"] = spec.model_id
    similarity_df["cs_length"] = spec.cs_length
    similarity_df["seed"] = spec.seed
    output_dir.mkdir(parents=True, exist_ok=True)
    similarity_df.to_csv(output_dir / "celltype_label_similarity.csv", index=False)
    return similarity_df


def _loaded_cell_embedding_frame(embeddings_dict: dict[str, dict[str, dict[str, Any]]]) -> pd.DataFrame:
    model_key = next(iter(embeddings_dict))
    loaded = ev.load_dataset_embedding_artifacts(
        embeddings_dict[model_key]["scrna"],
        annotation_column=ANNOTATION_COLUMN,
    )
    return loaded["artifacts"]["df_cells"]["dataframe"]


def append_embedding_to_adata(
    adata: ad.AnnData,
    spec: ModelSpec,
    embeddings_dict: dict[str, dict[str, dict[str, Any]]],
) -> ad.AnnData:
    df_cells = _loaded_cell_embedding_frame(embeddings_dict)
    embedding_columns = [
        column
        for column in df_cells.columns
        if isinstance(column, (int, np.integer))
        or (isinstance(column, str) and column.isdigit())
    ]
    embeddings = df_cells.loc[:, embedding_columns].copy()
    embeddings.index = embeddings.index.astype(str)
    return align_embedding_frame_to_adata(
        adata,
        embeddings,
        obsm_key=spec.model_id,
        copy=False,
        require_all_cells=True,
    )


def _load_scib_benchmark_classes():
    try:
        from scib_metrics.benchmark import Benchmarker, BioConservation
    except ImportError as error:
        raise ImportError(
            "scib_metrics is required for this script. Install it with:\n"
            'uv pip install --python .venv_annotation/bin/python "scib-metrics>=0.5.0"'
        ) from error
    return Benchmarker, BioConservation


def run_scib_bioconservation(
    adata: ad.AnnData,
    available_specs: list[ModelSpec],
) -> pd.DataFrame:
    if not available_specs:
        return pd.DataFrame(columns=["model_id", "cs_length", "seed", *SCIB_SCORE_COLUMNS])

    Benchmarker, BioConservation = _load_scib_benchmark_classes()
    bm = Benchmarker(
        adata,
        batch_key=BATCH_KEY,
        label_key=LABEL_KEY,
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=None,
        embedding_obsm_keys=[spec.model_id for spec in available_specs],
        n_jobs=N_JOBS,
    )
    bm.benchmark()
    SCIB_ROOT.mkdir(parents=True, exist_ok=True)
    raw_results = bm.get_results(min_max_scale=False)
    scaled_results = bm.get_results(min_max_scale=True)
    raw_results.to_csv(SCIB_ROOT / "scib_bioconservation_results_raw.csv")
    scaled_results.to_csv(SCIB_ROOT / "scib_bioconservation_results_minmax_scaled.csv")

    score_df = raw_results.drop(index="Metric Type", errors="ignore").reset_index(names="model_id")
    score_df = score_df.loc[:, ["model_id", *SCIB_SCORE_COLUMNS]]
    spec_df = pd.DataFrame(
        [
            {"model_id": spec.model_id, "cs_length": spec.cs_length, "seed": spec.seed}
            for spec in available_specs
        ]
    )
    return score_df.merge(spec_df, on="model_id", how="inner", validate="one_to_one")


def summarize_label_alignment(label_alignment_df: pd.DataFrame) -> pd.DataFrame:
    if label_alignment_df.empty:
        return pd.DataFrame(columns=["model_id", "cs_length", "seed", "mean_roc_auc", "sd_roc_auc", "n_cell_types"])
    return (
        label_alignment_df.groupby(["model_id", "cs_length", "seed"], as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            sd_roc_auc=("roc_auc", "std"),
            n_cell_types=("cell_type", "nunique"),
        )
    )


def build_model_summary(scib_df: pd.DataFrame, roc_summary: pd.DataFrame) -> pd.DataFrame:
    return scib_df.merge(
        roc_summary,
        on=["model_id", "cs_length", "seed"],
        how="outer",
        validate="one_to_one",
    )


def summarize_by_length(model_summary: pd.DataFrame) -> pd.DataFrame:
    if model_summary.empty:
        return pd.DataFrame()
    return (
        model_summary.groupby("cs_length", as_index=False)
        .agg(
            mean_total_scib=("Bio conservation", "mean"),
            sd_total_scib=("Bio conservation", "std"),
            n_scib_models=("Bio conservation", "count"),
            mean_roc_auc=("mean_roc_auc", "mean"),
            sd_roc_auc=("mean_roc_auc", "std"),
            n_roc_models=("mean_roc_auc", "count"),
        )
    )


def _save_figure(fig: plt.Figure, stem: str) -> None:
    save_publication_figure(fig, PLOTS_ROOT, stem, formats=("pdf", "png", "svg"))


def plot_scib_vs_roc_auc(model_summary: pd.DataFrame) -> None:
    plot_df = model_summary.dropna(subset=["Bio conservation", "mean_roc_auc"]).copy()
    if plot_df.empty:
        return
    palette = cs_length_palette(sorted(plot_df["cs_length"].astype(int).unique().tolist()))
    fig, ax = plt.subplots(figsize=(2.65, PLOT_HEIGHT))
    sns.scatterplot(
        data=plot_df,
        x="Bio conservation",
        y="mean_roc_auc",
        hue="cs_length",
        palette=palette,
        s=42,
        edgecolor=PBMC3K_SCATTER_EDGE_COLOR,
        linewidth=0.35,
        ax=ax,
    )
    ax.set_xlabel("scIB bio conservation score", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Mean ROC-AUC", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    legend = ax.legend(
        title="cs length",
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        borderaxespad=0.1,
    )
    if legend is not None:
        legend.get_title().set_fontsize(8)
    sns.despine(ax=ax)
    _save_figure(fig, "scib_bioconservation_vs_mean_roc_auc")


def _plot_length_bar(
    summary: pd.DataFrame,
    *,
    mean_column: str,
    sd_column: str,
    ylabel: str,
    stem: str,
) -> None:
    plot_df = summary.copy()
    if plot_df.empty:
        return
    plot_df["cs_length"] = pd.Categorical(plot_df["cs_length"], categories=CS_LENGTHS, ordered=True)
    plot_df = plot_df.sort_values("cs_length", kind="mergesort")
    palette = cs_length_palette([int(value) for value in plot_df["cs_length"]])
    x_positions = np.arange(len(plot_df)) * 0.52

    fig, ax = plt.subplots(figsize=(2.45, PLOT_HEIGHT))
    ax.bar(
        x_positions,
        plot_df[mean_column].astype(float),
        yerr=plot_df[sd_column].fillna(0).astype(float),
        width=BAR_WIDTH,
        color=[palette[int(length)] for length in plot_df["cs_length"]],
        edgecolor=BAR_EDGE_COLOR,
        linewidth=BAR_EDGE_LINEWIDTH,
        capsize=2.0,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(int(length)) for length in plot_df["cs_length"]], rotation=45, ha="right")
    ax.set_xlabel("Cell sentence length", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    sns.despine(ax=ax)
    _save_figure(fig, stem)


def plot_summary_bars(length_summary: pd.DataFrame) -> None:
    _plot_length_bar(
        length_summary,
        mean_column="mean_total_scib",
        sd_column="sd_total_scib",
        ylabel="Mean scIB bio conservation",
        stem="mean_scib_bioconservation_by_cs_length",
    )
    _plot_length_bar(
        length_summary,
        mean_column="mean_roc_auc",
        sd_column="sd_roc_auc",
        ylabel="Mean ROC-AUC",
        stem="mean_roc_auc_by_cs_length",
    )


def plot_roc_auc_per_celltype(label_alignment_df: pd.DataFrame) -> None:
    if label_alignment_df.empty:
        return
    plot_df = label_alignment_df.copy()
    plot_df["cs_length"] = pd.to_numeric(plot_df["cs_length"], errors="coerce")
    plot_df = plot_df.dropna(subset=["cs_length"])
    if plot_df.empty:
        return
    plot_df["cs_length"] = plot_df["cs_length"].astype(int)
    plot_df["cs_length"] = pd.Categorical(plot_df["cs_length"], categories=CS_LENGTHS, ordered=True)
    palette = cs_length_palette(CS_LENGTHS)
    fig, ax = plt.subplots(figsize=(2.65, PLOT_HEIGHT))
    sns.boxplot(
        data=plot_df,
        x="cs_length",
        y="roc_auc",
        hue="cs_length",
        order=CS_LENGTHS,
        hue_order=CS_LENGTHS,
        palette=palette,
        legend=False,
        width=0.45,
        showfliers=False,
        linewidth=0.65,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="cs_length",
        y="roc_auc",
        order=CS_LENGTHS,
        color=PBMC3K_CELLTYPE_POINT_COLOR,
        size=2.0,
        alpha=0.45,
        jitter=0.18,
        ax=ax,
    )
    ax.set_xlabel("Cell sentence length", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("ROC-AUC per cell type", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    sns.despine(ax=ax)
    _save_figure(fig, "roc_auc_per_celltype_by_cs_length")


def write_outputs(
    *,
    label_alignment_df: pd.DataFrame,
    scib_df: pd.DataFrame,
    model_summary: pd.DataFrame,
    length_summary: pd.DataFrame,
    selected_specs: list[ModelSpec],
    args: argparse.Namespace,
) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    label_alignment_df.to_csv(OUTPUT_ROOT / "pbmc3k_label_alignment_per_celltype.csv", index=False)
    scib_df.to_csv(OUTPUT_ROOT / "pbmc3k_scib_bioconservation_scores.csv", index=False)
    model_summary.to_csv(OUTPUT_ROOT / "pbmc3k_model_summary.csv", index=False)
    length_summary.to_csv(OUTPUT_ROOT / "pbmc3k_cs_length_summary.csv", index=False)
    metadata = {
        "study_id": STUDY_ID,
        "dataset_id": DATASET_ID,
        "annotation_column": ANNOTATION_COLUMN,
        "batch_key": BATCH_KEY,
        "scib_mode": "bio_conservation_only",
        "scib_score_columns": SCIB_SCORE_COLUMNS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "dataset_root": str(DATA_ROOT),
        "model_root": str(MODEL_ROOT),
        "model_sources": {spec.model_id: spec.model_reference for spec in selected_specs},
        "embedding_root": str(EMBEDDINGS_ROOT),
    }
    with (OUTPUT_ROOT / "pbmc3k_evaluation_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def load_existing_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    label_alignment_df = pd.read_csv(OUTPUT_ROOT / "pbmc3k_label_alignment_per_celltype.csv")
    scib_df = pd.read_csv(OUTPUT_ROOT / "pbmc3k_scib_bioconservation_scores.csv")
    model_summary = pd.read_csv(OUTPUT_ROOT / "pbmc3k_model_summary.csv")
    length_summary = pd.read_csv(OUTPUT_ROOT / "pbmc3k_cs_length_summary.csv")
    return label_alignment_df, scib_df, model_summary, length_summary


def run(args: argparse.Namespace) -> pd.DataFrame:
    set_publication_style()

    if args.plots_only:
        if "PBMC3K_EVALUATION_RUN_DIR" not in os.environ:
            raise ValueError("Set PBMC3K_EVALUATION_RUN_DIR to the existing analysis run for --plots-only.")
        PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
        label_alignment_df, _, model_summary, length_summary = load_existing_outputs()
        plot_scib_vs_roc_auc(model_summary)
        plot_summary_bars(length_summary)
        plot_roc_auc_per_celltype(label_alignment_df)
        print(f"Regenerated PBMC3k evaluation plots from existing outputs in {OUTPUT_ROOT}")
        return model_summary

    specs = select_model_specs(
        build_model_specs(), cs_length=args.cs_length, training_seed=args.training_seed
    )
    require_local_models(specs)
    PLOTS_ROOT.mkdir(parents=True, exist_ok=True)
    embedding_results: list[tuple[ModelSpec, dict[str, dict[str, dict[str, Any]]]]] = []
    label_frames: list[pd.DataFrame] = []

    for spec in specs:
        embeddings_dict = generate_model_embeddings(
            spec,
            force=args.force_embeddings,
            batch_size=args.batch_size,
            max_cells=args.max_cells,
        )
        label_frames.append(run_label_alignment(spec, embeddings_dict))
        embedding_results.append((spec, embeddings_dict))

    label_alignment_df = (
        pd.concat(label_frames, ignore_index=True)
        if label_frames
        else pd.DataFrame(columns=["model_id", "cs_length", "seed", "cell_type", "roc_auc"])
    )
    roc_summary = summarize_label_alignment(label_alignment_df)

    scib_df = pd.DataFrame(columns=["model_id", "cs_length", "seed", *SCIB_SCORE_COLUMNS])
    if embedding_results and not args.skip_scib:
        adata = _load_adata_test(embedding_results[0][0])
        for spec, embeddings_dict in embedding_results:
            adata = append_embedding_to_adata(adata, spec, embeddings_dict)
        scib_df = run_scib_bioconservation(adata, specs)
        del adata
        gc.collect()

    model_summary = build_model_summary(scib_df, roc_summary)
    length_summary = summarize_by_length(model_summary)
    write_outputs(
        label_alignment_df=label_alignment_df,
        scib_df=scib_df,
        model_summary=model_summary,
        length_summary=length_summary,
        selected_specs=specs,
        args=args,
    )
    plot_scib_vs_roc_auc(model_summary)
    plot_summary_bars(length_summary)
    plot_roc_auc_per_celltype(label_alignment_df)
    print(f"Saved PBMC3k evaluation outputs to {OUTPUT_ROOT}")
    return model_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PBMC3k cell sentence length stability models."
    )
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--skip-scib", action="store_true")
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate figures from saved aggregate CSVs without recomputing embeddings, label alignment, or scIB.",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-cells", type=int, default=DEFAULT_MAX_CELLS)
    parser.add_argument("--cs-length", help="One length, a comma-separated list, or all (default).")
    parser.add_argument("--training-seed", help="One seed, a comma-separated list, or all (default).")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
