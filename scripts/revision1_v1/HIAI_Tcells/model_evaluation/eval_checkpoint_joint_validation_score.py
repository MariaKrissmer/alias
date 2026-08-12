from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import Dataset, load_from_disk
from sentence_transformers import SentenceTransformer


DATASET_ID = "S2_heldout_donor_semantic_200k"
MODEL_CONFIGS = [
    {
        "short_name": "Base",
        "model_id": "PubMedBERTBase",
        "model_source": "neuml/pubmedbert-base-embeddings",
        "checkpoints": [{"checkpoint": "Base", "epoch": 0, "phase": "baseline"}],
        "color": sns.color_palette("tab20")[14],
    },
    {
        "short_name": "MI",
        "model_id": "MI_HIAI_Tcells_N1_200k_lr5e5",
        "dataset_sequence": ["ncbi"],
        "color": sns.color_palette("tab20")[2],
    },
    {
        "short_name": "MF",
        "model_id": "MF_HIAI_Tcells_S3_200k_lr5e5",
        "dataset_sequence": ["scrna"],
        "color": sns.color_palette("tab20")[0],
    },
    {
        "short_name": "MG",
        "model_id": "MG_HIAI_Tcells_S2_200k_lr5e5",
        "dataset_sequence": ["scrna"],
        "color": sns.color_palette("tab20")[4],
    },
    {
        "short_name": "MB",
        "model_id": "MB_HIAI_Tcells_S2_N1_200k_lr5e5",
        "dataset_sequence": ["ncbi", "scrna"],
        "color": sns.color_palette("tab20")[6],
    },
    {
        "short_name": "MD",
        "model_id": "MD_HIAI_Tcells_S4_N1_200k_lr5e5",
        "dataset_sequence": ["ncbi", "scrna"],
        "color": sns.color_palette("tab20")[8],
    },
]

SCRNA_EVAL_PATH = Path(
    os.environ.get(
        "JOINT_VALIDATION_SCRNA_EVAL_PATH",
        str(
            PROJECT_ROOT
            / "out"
            / "data"
            / "revision1_v1"
            / "HIAI_Tcells"
            / DATASET_ID
            / "datasets"
            / "scrna_eval_MNR_hnm"
        ),
    )
)
NCBI_EVAL_PATH = Path(
    os.environ.get(
        "JOINT_VALIDATION_NCBI_EVAL_PATH",
        str(
            PROJECT_ROOT
            / "out"
            / "data"
            / "revision1_v1"
            / "HIAI_Tcells"
            / "N1_ncbi_literature"
            / "datasets"
            / "ncbi_eval_MNR_hnm"
        ),
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "JOINT_VALIDATION_OUTPUT_DIR",
        str(
            PROJECT_ROOT
            / "out"
            / "data"
            / "revision1_v1"
            / "HIAI_Tcells"
            / DATASET_ID
            / "model_evaluation"
            / "checkpoint_joint_validation_score"
        ),
    )
)

MAX_EVAL_TRIPLETS = int(os.environ.get("JOINT_VALIDATION_MAX_EVAL_TRIPLETS", "1000"))
BATCH_SIZE = int(os.environ.get("JOINT_VALIDATION_BATCH_SIZE", "256"))
EMBEDDING_CACHE_DIR = Path(
    os.environ.get("JOINT_VALIDATION_EMBEDDING_CACHE_DIR", str(OUTPUT_DIR / "embedding_cache"))
)
REUSE_EMBEDDING_CACHE = (
    os.environ.get("JOINT_VALIDATION_REUSE_EMBEDDING_CACHE", "true").lower()
    not in {"0", "false", "no"}
)
FORCE_REGENERATE_EMBEDDING_CACHE = (
    os.environ.get("JOINT_VALIDATION_FORCE_REGENERATE_EMBEDDING_CACHE", "false").lower()
    in {"1", "true", "yes"}
)
ACCURACY_WEIGHT = float(os.environ.get("JOINT_VALIDATION_ACCURACY_WEIGHT", "0.7"))
RANKING_QUALITY_WEIGHT = float(
    os.environ.get("JOINT_VALIDATION_RANKING_QUALITY_WEIGHT", "0.3")
)
DATASET_WEIGHTS = {
    "scrna": float(os.environ.get("JOINT_VALIDATION_SCRNA_WEIGHT", "0.7")),
    "ncbi": float(os.environ.get("JOINT_VALIDATION_NCBI_WEIGHT", "0.3")),
}
CHECKPOINTS_ENV = os.environ.get("JOINT_VALIDATION_CHECKPOINTS", "").strip()

DATASET_VALIDATION_SCORE_FORMULA = (
    "dataset_validation_score = "
    f"{ACCURACY_WEIGHT:g} * triplet_accuracy + "
    f"{RANKING_QUALITY_WEIGHT:g} * ranking_quality"
)
JOINT_VALIDATION_SCORE_FORMULA = (
    "joint_validation_score = "
    f"{DATASET_WEIGHTS['scrna']:g} * scrna_dataset_validation_score + "
    f"{DATASET_WEIGHTS['ncbi']:g} * ncbi_dataset_validation_score"
)
COMBINED_PAIRWISE_RANKING_LOSS_FORMULA = (
    "combined_pairwise_ranking_loss = "
    f"{DATASET_WEIGHTS['scrna']:g} * scrna_pairwise_ranking_loss + "
    f"{DATASET_WEIGHTS['ncbi']:g} * ncbi_pairwise_ranking_loss"
)


def _model_root_candidates(model_id: str) -> list[Path]:
    root_names = [model_id]
    base_name = model_id.removesuffix("_all")
    root_names.extend(
        [
            f"{base_name}_e15_all",
            f"{base_name}_e8_all",
            f"{base_name}_all",
            base_name,
        ]
    )
    deduplicated_names = list(dict.fromkeys(root_names))
    return [PROJECT_ROOT / "models" / root_name for root_name in deduplicated_names]


def _model_root(model_config: dict) -> Path | None:
    env_name = f"{model_config['short_name']}_TCELLS_MODEL_ROOT"
    if os.environ.get(env_name):
        return Path(os.environ[env_name])

    for candidate in _model_root_candidates(model_config["model_id"]):
        if candidate.exists():
            return candidate
    return None


def _checkpoint_step(checkpoint_name: str) -> int:
    match = re.search(r"checkpoint-(\d+)", checkpoint_name)
    if match:
        return int(match.group(1))

    match = re.search(r"epoch_(\d+)", checkpoint_name)
    if match:
        return int(match.group(1))

    return 0


def _checkpoint_phase(checkpoint_name: str) -> str | None:
    match = re.search(r"epoch_\d+_(.+)$", checkpoint_name)
    if match:
        return match.group(1)
    if checkpoint_name.startswith("checkpoint-"):
        return "checkpoint"
    return None


def _checkpoint_mtime_ns(checkpoint_path: Path | str) -> int:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return 0
    return max(
        (path.stat().st_mtime_ns for path in checkpoint_path.rglob("*") if path.is_file()),
        default=checkpoint_path.stat().st_mtime_ns,
    )


def _dataset_text_hash(dataset: Dataset) -> str:
    digest = hashlib.sha256()
    digest.update(str(len(dataset)).encode("utf-8"))
    for column_name in ("sentence1", "sentence2", "negative"):
        digest.update(column_name.encode("utf-8"))
        for value in dataset[column_name]:
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _cache_safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _triplet_embedding_cache_dir(
    *,
    short_name: str,
    checkpoint_name: str,
    dataset_name: str,
) -> Path:
    return EMBEDDING_CACHE_DIR / short_name / _cache_safe_name(checkpoint_name) / dataset_name


def _triplet_embedding_cache_metadata(
    *,
    model_config: dict,
    dataset: Dataset,
    dataset_name: str,
    checkpoint_name: str,
    checkpoint_path: Path | str,
) -> dict:
    return {
        "short_name": model_config["short_name"],
        "model_id": model_config["model_id"],
        "checkpoint": checkpoint_name,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_mtime_ns": _checkpoint_mtime_ns(checkpoint_path),
        "dataset": dataset_name,
        "dataset_text_hash": _dataset_text_hash(dataset),
        "n_eval_triplets": len(dataset),
        "max_eval_triplets": MAX_EVAL_TRIPLETS,
        "batch_size": BATCH_SIZE,
        "normalize_embeddings": True,
    }


def _load_triplet_embedding_cache(
    cache_dir: Path,
    expected_metadata: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if FORCE_REGENERATE_EMBEDDING_CACHE or not REUSE_EMBEDDING_CACHE:
        return None

    metadata_path = cache_dir / "metadata.json"
    embeddings_path = cache_dir / "triplet_embeddings.npz"
    if not metadata_path.exists() or not embeddings_path.exists():
        return None

    with metadata_path.open("r", encoding="utf-8") as handle:
        stored_metadata = json.load(handle)

    if stored_metadata != expected_metadata:
        return None

    loaded = np.load(embeddings_path)
    return (
        loaded["anchor_embeddings"],
        loaded["positive_embeddings"],
        loaded["negative_embeddings"],
    )


def _save_triplet_embedding_cache(
    cache_dir: Path,
    *,
    metadata: dict,
    anchor_embeddings: np.ndarray,
    positive_embeddings: np.ndarray,
    negative_embeddings: np.ndarray,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_dir / "triplet_embeddings.npz",
        anchor_embeddings=anchor_embeddings,
        positive_embeddings=positive_embeddings,
        negative_embeddings=negative_embeddings,
    )
    with (cache_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _has_reusable_triplet_embedding_cache(
    *,
    model_config: dict,
    dataset: Dataset,
    dataset_name: str,
    checkpoint_name: str,
    checkpoint_path: Path | str,
) -> bool:
    metadata = _triplet_embedding_cache_metadata(
        model_config=model_config,
        dataset=dataset,
        dataset_name=dataset_name,
        checkpoint_name=checkpoint_name,
        checkpoint_path=checkpoint_path,
    )
    cache_dir = _triplet_embedding_cache_dir(
        short_name=model_config["short_name"],
        checkpoint_name=checkpoint_name,
        dataset_name=dataset_name,
    )
    return _load_triplet_embedding_cache(cache_dir, metadata) is not None


def _discover_checkpoints(model_config: dict) -> list[dict]:
    if "checkpoints" in model_config:
        return [
            {
                "short_name": model_config["short_name"],
                "model_id": model_config["model_id"],
                "checkpoint": checkpoint["checkpoint"],
                "checkpoint_epoch": checkpoint["epoch"],
                "checkpoint_phase": checkpoint["phase"],
                "model_source": model_config["model_source"],
            }
            for checkpoint in model_config["checkpoints"]
        ]

    root = _model_root(model_config)
    if root is None:
        print(
            f"Skipping {model_config['short_name']}: no model root found for "
            f"{model_config['model_id']}."
        )
        return []

    if CHECKPOINTS_ENV:
        checkpoint_names = [value.strip() for value in CHECKPOINTS_ENV.split(",") if value.strip()]
    else:
        checkpoint_names = [
            path.name
            for path in root.iterdir()
            if path.is_dir()
            and (path.name.startswith("checkpoint-") or path.name.startswith("epoch_"))
        ]

    checkpoint_names = sorted(checkpoint_names, key=lambda value: (_checkpoint_step(value), value))
    if not checkpoint_names:
        print(
            f"Skipping {model_config['short_name']}: no checkpoint folders found under {root}."
        )
        return []

    return [
        {
            "short_name": model_config["short_name"],
            "model_id": model_config["model_id"],
            "checkpoint": checkpoint_name,
            "checkpoint_epoch": _checkpoint_step(checkpoint_name),
            "checkpoint_phase": _checkpoint_phase(checkpoint_name),
            "model_source": root / checkpoint_name,
        }
        for checkpoint_name in checkpoint_names
    ]


def _load_eval_dataset(path: Path, dataset_name: str) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(f"{dataset_name} eval dataset not found: {path}")

    dataset = load_from_disk(str(path))
    required = {"sentence1", "sentence2", "negative"}
    missing = required.difference(dataset.column_names)
    if missing:
        raise ValueError(
            f"{dataset_name} eval dataset is missing required triplet columns: "
            f"{sorted(missing)}"
        )
    if MAX_EVAL_TRIPLETS and len(dataset) > MAX_EVAL_TRIPLETS:
        dataset = dataset.select(range(MAX_EVAL_TRIPLETS))
    return dataset


def _cosine_pairwise(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    left_norm[left_norm == 0] = 1.0
    right_norm[right_norm == 0] = 1.0
    return np.sum(left * right, axis=1) / (left_norm * right_norm)


def _encode(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    return np.asarray(
        model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    )


def _evaluate_triplets(
    *,
    model_config: dict,
    model: SentenceTransformer | None,
    dataset: Dataset,
    dataset_name: str,
    checkpoint_name: str,
    checkpoint_path: Path | str,
    checkpoint_epoch: int,
    checkpoint_phase: str | None,
) -> dict:
    anchors = [str(value) for value in dataset["sentence1"]]
    positives = [str(value) for value in dataset["sentence2"]]
    negatives = [str(value) for value in dataset["negative"]]

    cache_dir = _triplet_embedding_cache_dir(
        short_name=model_config["short_name"],
        checkpoint_name=checkpoint_name,
        dataset_name=dataset_name,
    )
    cache_metadata = _triplet_embedding_cache_metadata(
        model_config=model_config,
        dataset=dataset,
        dataset_name=dataset_name,
        checkpoint_name=checkpoint_name,
        checkpoint_path=checkpoint_path,
    )
    cached_embeddings = _load_triplet_embedding_cache(cache_dir, cache_metadata)

    if cached_embeddings is not None:
        print(f"Reusing cached embeddings for {model_config['short_name']} {checkpoint_name} on {dataset_name}")
        anchor_embeddings, positive_embeddings, negative_embeddings = cached_embeddings
    else:
        if model is None:
            raise ValueError(
                f"Missing reusable embeddings for {model_config['short_name']} {checkpoint_name} "
                f"on {dataset_name}, but no model was loaded."
            )
        anchor_embeddings = _encode(model, anchors)
        positive_embeddings = _encode(model, positives)
        negative_embeddings = _encode(model, negatives)
        _save_triplet_embedding_cache(
            cache_dir,
            metadata=cache_metadata,
            anchor_embeddings=anchor_embeddings,
            positive_embeddings=positive_embeddings,
            negative_embeddings=negative_embeddings,
        )

    positive_similarity = _cosine_pairwise(anchor_embeddings, positive_embeddings)
    negative_similarity = _cosine_pairwise(anchor_embeddings, negative_embeddings)
    margins = positive_similarity - negative_similarity

    return {
        "short_name": model_config["short_name"],
        "model_id": model_config["model_id"],
        "checkpoint": checkpoint_name,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_phase": checkpoint_phase,
        "checkpoint_path": str(checkpoint_path),
        "dataset": dataset_name,
        "dataset_weight": DATASET_WEIGHTS[dataset_name],
        "n_eval_triplets": len(dataset),
        "triplet_accuracy": float(np.mean(margins > 0)),
        "pairwise_ranking_loss": float(np.logaddexp(0.0, -margins).mean()),
        "mean_positive_similarity": float(positive_similarity.mean()),
        "mean_negative_similarity": float(negative_similarity.mean()),
        "mean_similarity_margin": float(margins.mean()),
        "median_similarity_margin": float(np.median(margins)),
    }


def _add_dataset_normalized_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    metrics["ranking_quality"] = np.nan

    for dataset_name, dataset_rows in metrics.groupby("dataset"):
        loss = dataset_rows["pairwise_ranking_loss"]
        loss_min = float(loss.min())
        loss_max = float(loss.max())
        if np.isclose(loss_max, loss_min):
            quality = pd.Series(1.0, index=dataset_rows.index)
        else:
            quality = 1.0 - ((loss - loss_min) / (loss_max - loss_min))
        metrics.loc[dataset_rows.index, "ranking_quality"] = quality

    metrics["dataset_validation_score"] = (
        ACCURACY_WEIGHT * metrics["triplet_accuracy"]
        + RANKING_QUALITY_WEIGHT * metrics["ranking_quality"]
    )
    return metrics


def _checkpoint_scores(metrics: pd.DataFrame) -> pd.DataFrame:
    weighted = metrics.copy()
    total_weight = weighted.groupby(["short_name", "checkpoint"])["dataset_weight"].transform("sum")
    weighted["normalized_dataset_weight"] = weighted["dataset_weight"] / total_weight

    score_summary = (
        weighted.assign(
            weighted_triplet_accuracy=lambda df: (
                df["triplet_accuracy"] * df["normalized_dataset_weight"]
            ),
            weighted_ranking_quality=lambda df: (
                df["ranking_quality"] * df["normalized_dataset_weight"]
            ),
            weighted_pairwise_ranking_loss=lambda df: (
                df["pairwise_ranking_loss"] * df["normalized_dataset_weight"]
            ),
            weighted_validation_score=lambda df: (
                df["dataset_validation_score"] * df["normalized_dataset_weight"]
            ),
        )
        .groupby(
            [
                "short_name",
                "model_id",
                "checkpoint",
                "checkpoint_epoch",
                "checkpoint_phase",
                "checkpoint_path",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            joint_triplet_accuracy=("weighted_triplet_accuracy", "sum"),
            joint_ranking_quality=("weighted_ranking_quality", "sum"),
            combined_pairwise_ranking_loss=("weighted_pairwise_ranking_loss", "sum"),
            joint_validation_score=("weighted_validation_score", "sum"),
            n_datasets=("dataset", "nunique"),
        )
        .sort_values(["short_name", "checkpoint_epoch", "checkpoint"])
    )
    score_summary["joint_pairwise_ranking_loss"] = score_summary[
        "combined_pairwise_ranking_loss"
    ]

    loss = score_summary["combined_pairwise_ranking_loss"]
    loss_min = float(loss.min())
    loss_max = float(loss.max())
    if np.isclose(loss_max, loss_min):
        score_summary["combined_pairwise_ranking_loss_score"] = 1.0
    else:
        score_summary["combined_pairwise_ranking_loss_score"] = (
            1.0 - ((loss - loss_min) / (loss_max - loss_min))
        )
    score_summary["rank_by_combined_pairwise_ranking_loss"] = (
        score_summary["combined_pairwise_ranking_loss"]
        .rank(method="dense", ascending=True)
        .astype(int)
    )
    score_summary["rank_by_joint_validation_score"] = (
        score_summary["joint_validation_score"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    return score_summary


def _plot_checkpoint_metric(
    scores: pd.DataFrame,
    *,
    output_dir: Path,
    model_label: str,
    color: tuple[float, float, float],
    metric: str,
    ylabel: str,
    file_stem: str,
    highlight_best: bool = False,
) -> None:
    if scores.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_df = scores.sort_values(["checkpoint_epoch", "checkpoint"])
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.plot(
        plot_df["checkpoint_epoch"],
        plot_df[metric],
        marker="o",
        color=color,
        linewidth=1.25,
        markersize=4,
    )
    if highlight_best and not plot_df.empty:
        best = plot_df.loc[plot_df[metric].idxmax()]
        ax.scatter(
            [best["checkpoint_epoch"]],
            [best[metric]],
            color="black",
            s=35,
            zorder=3,
            label=f"best: {best['checkpoint']}",
        )
        ax.legend(frameon=False, loc="best")
    ax.set_xlabel("Checkpoint epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{model_label}: {ylabel}")
    sns.despine(top=True, right=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{file_stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{file_stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _plot_dataset_metric(
    metrics: pd.DataFrame,
    *,
    output_dir: Path,
    model_label: str,
    metric: str,
    ylabel: str,
    file_stem: str,
) -> None:
    if metrics.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_df = metrics.sort_values(["checkpoint_epoch", "dataset"])
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    sns.lineplot(
        data=plot_df,
        x="checkpoint_epoch",
        y=metric,
        hue="dataset",
        marker="o",
        palette={"scrna": sns.color_palette("tab20")[0], "ncbi": sns.color_palette("tab20")[2]},
        ax=ax,
    )
    ax.set_xlabel("Checkpoint epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{model_label}: {ylabel}")
    ax.legend(frameon=False, loc="best")
    sns.despine(top=True, right=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{file_stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{file_stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _same_phase_dataset_validation_progress(metrics: pd.DataFrame) -> pd.DataFrame:
    score_cols = [
        "dataset_validation_score",
        "triplet_accuracy",
        "pairwise_ranking_loss",
        "ranking_quality",
    ]
    progress = metrics[
        [
            "short_name",
            "checkpoint",
            "checkpoint_epoch",
            "checkpoint_phase",
            "dataset",
            *score_cols,
        ]
    ].copy()
    progress["previous_checkpoint_epoch"] = progress["checkpoint_epoch"] - 2

    previous = metrics[
        ["short_name", "checkpoint", "checkpoint_epoch", "dataset", *score_cols]
    ].rename(
        columns={
            "checkpoint": "previous_checkpoint",
            "checkpoint_epoch": "previous_checkpoint_epoch",
            **{column: f"previous_{column}" for column in score_cols},
        }
    )

    progress = progress.merge(
        previous,
        on=["short_name", "previous_checkpoint_epoch", "dataset"],
        how="left",
    )
    progress = progress[progress["previous_checkpoint"].notna()].copy()

    for column in score_cols:
        progress[f"delta_{column}"] = (
            progress[column] - progress[f"previous_{column}"]
        )

    return progress.sort_values(["short_name", "checkpoint_epoch", "dataset"])


def _plot_same_phase_dataset_validation_progress(
    progress: pd.DataFrame,
    *,
    output_dir: Path,
    model_label: str,
) -> None:
    if progress.empty:
        print("Skipping same-phase progress plot: no comparable checkpoints.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    sns.lineplot(
        data=progress,
        x="checkpoint_epoch",
        y="delta_dataset_validation_score",
        hue="dataset",
        marker="o",
        palette={"scrna": sns.color_palette("tab20")[0], "ncbi": sns.color_palette("tab20")[2]},
        ax=ax,
    )
    ax.axhline(0, color="0.35", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Checkpoint epoch")
    ax.set_ylabel("Delta dataset validation score")
    ax.set_title(f"{model_label}: progress vs previous comparable checkpoint")
    ax.legend(frameon=False, loc="best")
    sns.despine(top=True, right=True)
    fig.tight_layout()
    fig.savefig(
        output_dir / "same_phase_dataset_validation_score_delta.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "same_phase_dataset_validation_score_delta.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)


def _write_model_outputs(
    *,
    model_config: dict,
    metrics: pd.DataFrame,
    scores: pd.DataFrame,
    progress: pd.DataFrame,
) -> None:
    short_name = model_config["short_name"]
    output_dir = OUTPUT_DIR / short_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model_metrics = metrics[metrics["short_name"] == short_name].copy()
    model_scores = scores[scores["short_name"] == short_name].copy()
    model_progress = progress[progress["short_name"] == short_name].copy()
    if model_metrics.empty or model_scores.empty:
        return

    model_metrics.to_csv(output_dir / "per_dataset_validation_metrics.csv", index=False)
    model_scores.to_csv(output_dir / "checkpoint_selection_scores.csv", index=False)
    model_progress.to_csv(
        output_dir / "same_phase_dataset_validation_progress.csv",
        index=False,
    )

    _plot_checkpoint_metric(
        model_scores,
        output_dir=output_dir,
        model_label=short_name,
        color=model_config["color"],
        metric="joint_validation_score",
        ylabel="Joint validation score",
        file_stem="joint_validation_score_by_checkpoint",
        highlight_best=True,
    )
    _plot_checkpoint_metric(
        model_scores,
        output_dir=output_dir,
        model_label=short_name,
        color=model_config["color"],
        metric="joint_ranking_quality",
        ylabel="Joint ranking quality",
        file_stem="joint_ranking_quality_by_checkpoint",
    )
    _plot_checkpoint_metric(
        model_scores,
        output_dir=output_dir,
        model_label=short_name,
        color=model_config["color"],
        metric="combined_pairwise_ranking_loss",
        ylabel="Combined pairwise ranking loss",
        file_stem="combined_pairwise_ranking_loss_by_checkpoint",
    )
    _plot_checkpoint_metric(
        model_scores,
        output_dir=output_dir,
        model_label=short_name,
        color=model_config["color"],
        metric="combined_pairwise_ranking_loss_score",
        ylabel="Combined pairwise ranking loss score",
        file_stem="combined_pairwise_ranking_loss_score_by_checkpoint",
        highlight_best=True,
    )
    _plot_dataset_metric(
        model_metrics,
        output_dir=output_dir,
        model_label=short_name,
        metric="triplet_accuracy",
        ylabel="Triplet accuracy",
        file_stem="triplet_accuracy_by_dataset",
    )
    _plot_dataset_metric(
        model_metrics,
        output_dir=output_dir,
        model_label=short_name,
        metric="pairwise_ranking_loss",
        ylabel="Pairwise ranking loss",
        file_stem="pairwise_ranking_loss_by_dataset",
    )
    _plot_dataset_metric(
        model_metrics,
        output_dir=output_dir,
        model_label=short_name,
        metric="ranking_quality",
        ylabel="Ranking quality",
        file_stem="ranking_quality_by_dataset",
    )
    _plot_dataset_metric(
        model_metrics,
        output_dir=output_dir,
        model_label=short_name,
        metric="dataset_validation_score",
        ylabel="Dataset validation score",
        file_stem="dataset_validation_score_by_dataset",
    )
    _plot_same_phase_dataset_validation_progress(
        model_progress,
        output_dir=output_dir,
        model_label=short_name,
    )


def _write_best_checkpoints(scores: pd.DataFrame, metrics: pd.DataFrame) -> Path:
    best_rows = (
        scores.sort_values(
            [
                "short_name",
                "joint_validation_score",
                "joint_triplet_accuracy",
                "checkpoint_epoch",
            ],
            ascending=[True, False, False, True],
        )
        .groupby("short_name", as_index=False)
        .head(1)
    )
    payload = {
        "selection_rule": (
            "Within each model, maximize joint_validation_score; ties prefer higher "
            "joint_triplet_accuracy and then earlier checkpoint."
        ),
        "score_weights": {
            "accuracy_weight": ACCURACY_WEIGHT,
            "ranking_quality_weight": RANKING_QUALITY_WEIGHT,
            "dataset_weights": DATASET_WEIGHTS,
        },
        "score_formulas": {
            "dataset_validation_score": DATASET_VALIDATION_SCORE_FORMULA,
            "joint_validation_score": JOINT_VALIDATION_SCORE_FORMULA,
            "combined_pairwise_ranking_loss": COMBINED_PAIRWISE_RANKING_LOSS_FORMULA,
            "combined_pairwise_ranking_loss_score": (
                "combined_pairwise_ranking_loss_score = 1 - minmax("
                "combined_pairwise_ranking_loss) across all evaluated model checkpoints"
            ),
        },
        "best_checkpoints": [],
    }

    for _, best in best_rows.iterrows():
        best_metrics = metrics[
            (metrics["short_name"] == best["short_name"])
            & (metrics["checkpoint"] == best["checkpoint"])
        ]
        payload["best_checkpoints"].append(
            {
                "short_name": best["short_name"],
                "model_id": best["model_id"],
                "best_checkpoint": best["checkpoint"],
                "best_checkpoint_path": best["checkpoint_path"],
                "joint_validation_score": float(best["joint_validation_score"]),
                "joint_triplet_accuracy": float(best["joint_triplet_accuracy"]),
                "combined_pairwise_ranking_loss": float(best["combined_pairwise_ranking_loss"]),
                "combined_pairwise_ranking_loss_score": float(
                    best["combined_pairwise_ranking_loss_score"]
                ),
                "datasets": best_metrics[
                    [
                        "dataset",
                        "triplet_accuracy",
                        "pairwise_ranking_loss",
                        "ranking_quality",
                        "dataset_validation_score",
                        "n_eval_triplets",
                    ]
                ].to_dict(orient="records"),
            }
        )

    path = OUTPUT_DIR / "best_checkpoints_by_model.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_datasets = {
        "scrna": _load_eval_dataset(SCRNA_EVAL_PATH, "scrna"),
        "ncbi": _load_eval_dataset(NCBI_EVAL_PATH, "ncbi"),
    }
    candidates = [
        candidate
        for model_config in MODEL_CONFIGS
        for candidate in _discover_checkpoints(model_config)
    ]
    if not candidates:
        raise FileNotFoundError("No model candidates were found.")

    print("Evaluating multi-model joint validation score")
    print(f"scRNA eval: {SCRNA_EVAL_PATH} ({len(eval_datasets['scrna'])} triplets)")
    print(f"NCBI eval: {NCBI_EVAL_PATH} ({len(eval_datasets['ncbi'])} triplets)")
    print(DATASET_VALIDATION_SCORE_FORMULA)
    print(JOINT_VALIDATION_SCORE_FORMULA)
    print(COMBINED_PAIRWISE_RANKING_LOSS_FORMULA)
    print(f"Embedding cache: {EMBEDDING_CACHE_DIR}")

    rows = []
    for candidate in candidates:
        model_config = next(
            config for config in MODEL_CONFIGS if config["short_name"] == candidate["short_name"]
        )
        checkpoint = candidate["checkpoint"]
        checkpoint_path = candidate["model_source"]
        needs_model = any(
            not _has_reusable_triplet_embedding_cache(
                model_config=model_config,
                dataset=dataset,
                dataset_name=dataset_name,
                checkpoint_name=checkpoint,
                checkpoint_path=checkpoint_path,
            )
            for dataset_name, dataset in eval_datasets.items()
        )

        model = None
        if needs_model:
            print(f"Loading {candidate['short_name']} {checkpoint}: {checkpoint_path}")
            model = SentenceTransformer(str(checkpoint_path))
        for dataset_name, dataset in eval_datasets.items():
            print(f"Evaluating {candidate['short_name']} {checkpoint} on {dataset_name}")
            rows.append(
                _evaluate_triplets(
                    model_config=model_config,
                    model=model,
                    dataset=dataset,
                    dataset_name=dataset_name,
                    checkpoint_name=checkpoint,
                    checkpoint_path=checkpoint_path,
                    checkpoint_epoch=candidate["checkpoint_epoch"],
                    checkpoint_phase=candidate["checkpoint_phase"],
                )
            )
        if model is not None:
            del model

    metrics = _add_dataset_normalized_scores(pd.DataFrame(rows))
    scores = _checkpoint_scores(metrics)
    progress = _same_phase_dataset_validation_progress(metrics)

    metrics_path = OUTPUT_DIR / "per_dataset_validation_metrics.csv"
    scores_path = OUTPUT_DIR / "checkpoint_selection_scores.csv"
    progress_path = OUTPUT_DIR / "same_phase_dataset_validation_progress.csv"
    metrics.to_csv(metrics_path, index=False)
    scores.to_csv(scores_path, index=False)
    progress.to_csv(progress_path, index=False)
    best_path = _write_best_checkpoints(scores, metrics)

    for model_config in MODEL_CONFIGS:
        _write_model_outputs(
            model_config=model_config,
            metrics=metrics,
            scores=scores,
            progress=progress,
        )

    print(f"Saved per-dataset validation metrics to {metrics_path}")
    print(f"Saved checkpoint selection scores to {scores_path}")
    print(f"Saved same-phase dataset validation progress to {progress_path}")
    print(f"Saved best checkpoint metadata to {best_path}")


if __name__ == "__main__":
    main()
