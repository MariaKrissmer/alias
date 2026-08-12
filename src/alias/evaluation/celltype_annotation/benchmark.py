from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


REQUIRED_PREDICTION_FILE = "predictions.csv"
REQUIRED_EVALUATION_FILES = (
    "metrics_summary.csv",
    "metrics_summary.json",
    "per_celltype_metrics.csv",
    "confusion_matrix.csv",
    "confusion_matrix.pdf",
    "confusion_matrix.png",
    "metadata.json",
)

CacheStatus = Literal["complete", "missing_evaluation", "missing_predictions"]


def get_annotation_cache_status(model_dir: Path | str) -> CacheStatus:
    """Return whether annotation predictions/evaluation artifacts are reusable."""
    model_path = Path(model_dir)
    if not (model_path / REQUIRED_PREDICTION_FILE).exists():
        return "missing_predictions"

    if all((model_path / file_name).exists() for file_name in REQUIRED_EVALUATION_FILES):
        return "complete"

    return "missing_evaluation"


def _required_prediction_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    missing = {"true_label", "predicted_label"} - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing required columns: {sorted(missing)}")

    clean = predictions.copy()
    clean["true_label"] = clean["true_label"].astype(str)
    clean["predicted_label"] = clean["predicted_label"].astype(str)
    return clean


def compute_annotation_metrics(predictions: pd.DataFrame, model_name: str, dataset_id: str) -> dict[str, Any]:
    """Compute summary and per-celltype annotation metrics."""
    clean = _required_prediction_columns(predictions)
    y_true = clean["true_label"].to_numpy()
    y_pred = clean["predicted_label"].to_numpy()
    labels = sorted(set(y_true) | set(y_pred))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    per_celltype = pd.DataFrame(
        [
            {
                "cell_type": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            }
            for label in labels
        ]
    )

    summary = {
        "model_name": model_name,
        "dataset_id": dataset_id,
        "n_cells": int(len(clean)),
        "n_true_labels": int(len(set(y_true))),
        "n_predicted_labels": int(len(set(y_pred))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    confusion = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=pd.Index(labels, name="true_label"),
        columns=pd.Index(labels, name="predicted_label"),
    )

    return {
        "summary": summary,
        "per_celltype": per_celltype,
        "confusion": confusion,
    }


def _plot_confusion_matrix(confusion: pd.DataFrame, output_path: Path, title: str) -> None:
    n_labels = max(1, len(confusion))
    fig_width = max(7.0, min(24.0, 0.42 * n_labels + 3.0))
    fig_height = max(6.0, min(22.0, 0.38 * n_labels + 2.5))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(confusion.values, cmap="Blues", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(len(confusion.columns)))
    ax.set_yticks(np.arange(len(confusion.index)))
    ax.set_xticklabels(confusion.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(confusion.index, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_annotation_evaluation(
    predictions: pd.DataFrame,
    output_dir: Path | str,
    model_name: str,
    dataset_id: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write predictions, metrics, confusion matrix, plots, and metadata."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    clean = _required_prediction_columns(predictions)
    clean.index.name = clean.index.name or "cell_id"
    prediction_path = output_path / REQUIRED_PREDICTION_FILE
    clean.to_csv(prediction_path)

    metrics = compute_annotation_metrics(clean, model_name=model_name, dataset_id=dataset_id)
    summary_df = pd.DataFrame([metrics["summary"]])
    per_celltype = metrics["per_celltype"]
    confusion = metrics["confusion"]

    metrics_summary_csv = output_path / "metrics_summary.csv"
    metrics_summary_json = output_path / "metrics_summary.json"
    per_celltype_csv = output_path / "per_celltype_metrics.csv"
    confusion_csv = output_path / "confusion_matrix.csv"
    confusion_pdf = output_path / "confusion_matrix.pdf"
    confusion_png = output_path / "confusion_matrix.png"
    metadata_json = output_path / "metadata.json"

    summary_df.to_csv(metrics_summary_csv, index=False)
    with metrics_summary_json.open("w", encoding="utf-8") as handle:
        json.dump(metrics["summary"], handle, indent=2, sort_keys=True)
    per_celltype.to_csv(per_celltype_csv, index=False)
    confusion.to_csv(confusion_csv)
    _plot_confusion_matrix(confusion, confusion_pdf, title=f"{model_name} confusion matrix")
    _plot_confusion_matrix(confusion, confusion_png, title=f"{model_name} confusion matrix")

    combined_metadata = {
        "model_name": model_name,
        "dataset_id": dataset_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prediction_path": str(prediction_path),
        "metrics_summary_path": str(metrics_summary_csv),
        "per_celltype_metrics_path": str(per_celltype_csv),
        "confusion_matrix_path": str(confusion_csv),
    }
    if metadata:
        combined_metadata.update(metadata)
    with metadata_json.open("w", encoding="utf-8") as handle:
        json.dump(combined_metadata, handle, indent=2, sort_keys=True)

    return {
        "predictions_csv": prediction_path,
        "metrics_summary_csv": metrics_summary_csv,
        "metrics_summary_json": metrics_summary_json,
        "per_celltype_metrics_csv": per_celltype_csv,
        "confusion_matrix_csv": confusion_csv,
        "confusion_matrix_pdf": confusion_pdf,
        "confusion_matrix_png": confusion_png,
        "metadata_json": metadata_json,
    }


def _numeric_embedding_values(df: pd.DataFrame, excluded_columns: set[str]) -> np.ndarray:
    numeric_columns = [column for column in df.columns if column not in excluded_columns]
    numeric_df = df[numeric_columns].select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric embedding columns found.")
    return numeric_df.to_numpy(dtype=np.float32)


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True).T
    denominator = np.maximum(left_norm * right_norm, np.finfo(np.float32).eps)
    return (left @ right.T) / denominator


def make_similarity_top_label_predictions(
    cell_embeddings: pd.DataFrame,
    label_embeddings: pd.DataFrame,
    annotation_column: str,
    label_column: str = "cell_type",
) -> pd.DataFrame:
    """Assign each cell to the most similar embedded cell-type label."""
    if annotation_column not in cell_embeddings.columns:
        raise ValueError(f"Cell embeddings do not contain annotation column {annotation_column!r}.")
    if label_column not in label_embeddings.columns:
        raise ValueError(f"Label embeddings do not contain label column {label_column!r}.")

    cell_matrix = _numeric_embedding_values(cell_embeddings, excluded_columns={annotation_column})
    label_matrix = _numeric_embedding_values(label_embeddings, excluded_columns={label_column})
    labels = label_embeddings[label_column].astype(str).to_numpy()

    similarity = _cosine_similarity(cell_matrix, label_matrix)
    top_indices = np.argmax(similarity, axis=1)
    top_scores = similarity[np.arange(similarity.shape[0]), top_indices]

    return pd.DataFrame(
        {
            "true_label": cell_embeddings[annotation_column].astype(str).to_numpy(),
            "predicted_label": labels[top_indices],
            "score": top_scores.astype(float),
        },
        index=cell_embeddings.index,
    )


def make_similarity_predictions_from_embedding_artifacts(
    embeddings_dict: dict[str, dict[str, dict[str, Any]]],
    annotation_column: str,
    model_key: str | None = None,
    dataset_key: str = "scrna",
) -> pd.DataFrame:
    """Load saved embedding artifacts and convert similarities to top-label predictions."""
    from alias.evaluation.embedding import load_dataset_embedding_artifacts

    selected_model_key = model_key or next(iter(embeddings_dict))
    dataset_meta = embeddings_dict[selected_model_key][dataset_key]
    loaded = load_dataset_embedding_artifacts(dataset_meta, annotation_column=annotation_column)
    artifacts = loaded["artifacts"]

    cell_df = artifacts["df_cells"]["dataframe"]
    label_df = artifacts["df_celltypes"]["dataframe"]
    label_column = annotation_column if annotation_column in label_df.columns else "cell_type"

    return make_similarity_top_label_predictions(
        cell_embeddings=cell_df,
        label_embeddings=label_df,
        annotation_column=annotation_column,
        label_column=label_column,
    )
