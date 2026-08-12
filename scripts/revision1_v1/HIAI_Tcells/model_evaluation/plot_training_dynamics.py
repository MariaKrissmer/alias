from __future__ import annotations

import logging
import os
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import seaborn as sns


MODEL_CONFIGS = [
    {
        "short_name": "Base",
        "model_id": "neuml/pubmedbert-base-embeddings",
        "label": "Base",
        "dataset_sequence": [],
        "color": sns.color_palette("tab20")[14],
        "has_training_dynamics": False,
    },
    {
        "short_name": "MI",
        "model_id": "MI_HIAI_Tcells_N1_200k_lr5e5",
        "label": "MI",
        "dataset_sequence": ["ncbi"],
        "color": sns.color_palette("tab20")[2],
    },
    {
        "short_name": "MF",
        "model_id": "MF_HIAI_Tcells_S3_200k_lr5e5",
        "label": "MF",
        "dataset_sequence": ["scrna"],
        "color": sns.color_palette("tab20")[0],
    },
    {
        "short_name": "MG",
        "model_id": "MG_HIAI_Tcells_S2_200k_lr5e5",
        "label": "MG",
        "dataset_sequence": ["scrna"],
        "color": sns.color_palette("tab20")[4],
    },
    {
        "short_name": "MB",
        "model_id": "MB_HIAI_Tcells_S2_N1_200k_lr5e5",
        "label": "MB",
        "dataset_sequence": ["ncbi", "scrna"],
        "color": sns.color_palette("tab20")[6],
    },
    {
        "short_name": "MD",
        "model_id": "MD_HIAI_Tcells_S4_N1_200k_lr5e5",
        "label": "MD",
        "dataset_sequence": ["ncbi", "scrna"],
        "color": sns.color_palette("tab20")[8],
    },
]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "out"
    / "data"
    / "revision1_v1"
    / "HIAI_Tcells"
    / "model_evaluation"
    / "training_dynamics"
)


def _model_root_candidates(model_id: str) -> list[Path]:
    if "/" in model_id:
        return []

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


def _training_progress_path(model_config: dict) -> Path | None:
    model_root = _model_root(model_config)
    if model_root is None:
        return None
    return model_root / "metadata" / "training_progress.csv"


def _add_global_training_axis(
    progress: pd.DataFrame,
    *,
    dataset_sequence: list[str],
) -> pd.DataFrame:
    progress = progress.copy()
    phase_indices = []
    phase_index = 1
    previous_step = None

    for _, row in progress.iterrows():
        step = row.get("step")
        has_train_runtime = pd.notna(row.get("train_runtime"))

        if previous_step is not None and pd.notna(step) and step < previous_step:
            phase_index += 1

        phase_indices.append(phase_index)

        if has_train_runtime:
            phase_index += 1
            previous_step = None
        elif pd.notna(step):
            previous_step = step

    progress["phase_index"] = phase_indices
    if dataset_sequence:
        progress["phase_dataset"] = [
            dataset_sequence[(phase - 1) % len(dataset_sequence)]
            for phase in progress["phase_index"]
        ]
    else:
        progress["phase_dataset"] = "baseline"
    progress["global_epoch"] = (
        progress["phase_index"].astype(float)
        - 1.0
        + pd.to_numeric(progress["epoch"], errors="coerce").fillna(0.0)
    )
    return progress


def _load_model_progress(model_config: dict) -> pd.DataFrame | None:
    if not model_config.get("has_training_dynamics", True):
        return None

    progress_path = _training_progress_path(model_config)
    if progress_path is None or not progress_path.exists():
        print(
            "Skipping training dynamics for "
            f"{model_config['short_name']}: no training_progress.csv found."
        )
        return None

    progress = pd.read_csv(progress_path)
    progress = _add_global_training_axis(
        progress,
        dataset_sequence=model_config["dataset_sequence"],
    )
    progress["short_name"] = model_config["short_name"]
    progress["model_id"] = model_config["model_id"]
    progress["model_label"] = model_config["label"]
    progress["model_root"] = str(progress_path.parents[1])
    return progress


def _training_rows(progress: pd.DataFrame) -> pd.DataFrame:
    return progress[pd.notna(progress["loss"])].copy()


def _eval_rows(progress: pd.DataFrame) -> pd.DataFrame:
    return progress[pd.notna(progress["eval_loss"])].copy()


def _model_output_dir(model_config: dict) -> Path:
    return OUTPUT_DIR / model_config["short_name"]


def _plot_metric(
    *,
    progress: pd.DataFrame,
    model_config: dict,
    metric: str,
    ylabel: str,
    output_name: str,
    include_eval_loss: bool = False,
) -> None:
    output_dir = _model_output_dir(model_config)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_rows = _training_rows(
        progress[progress["short_name"] == model_config["short_name"]]
    )
    if model_rows.empty or metric not in model_rows or model_rows[metric].dropna().empty:
        return

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(
        model_rows["global_epoch"],
        model_rows[metric],
        label=model_config["label"],
        color=model_config["color"],
        linewidth=1.25,
    )

    if include_eval_loss:
        eval_rows = _eval_rows(progress[progress["short_name"] == model_config["short_name"]])
        if not eval_rows.empty:
            ax.scatter(
                eval_rows["global_epoch"],
                eval_rows["eval_loss"],
                color=model_config["color"],
                edgecolor="black",
                linewidth=0.35,
                s=18,
                marker="o",
                zorder=3,
                label="eval_loss",
            )

    ax.set_xlabel("Training phase")
    ax.set_ylabel(ylabel)
    ax.set_title(model_config["label"])
    ax.legend(frameon=False, loc="best")
    sns.despine(top=True, right=True)
    fig.tight_layout()
    fig.savefig(output_dir / f"{output_name}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{output_name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _write_model_training_dynamics(progress: pd.DataFrame, model_config: dict) -> None:
    output_dir = _model_output_dir(model_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_progress = progress[progress["short_name"] == model_config["short_name"]].copy()
    if model_progress.empty:
        return

    model_progress.to_csv(output_dir / "training_progress_with_global_axis.csv", index=False)
    _plot_metric(
        progress=model_progress,
        model_config=model_config,
        metric="loss",
        ylabel="Training loss",
        output_name="training_loss",
        include_eval_loss=True,
    )
    _plot_metric(
        progress=model_progress,
        model_config=model_config,
        metric="learning_rate",
        ylabel="Learning rate",
        output_name="learning_rate",
    )
    _plot_metric(
        progress=model_progress,
        model_config=model_config,
        metric="grad_norm",
        ylabel="Logged grad norm",
        output_name="grad_norm",
    )


def _plot_available_models_summary(progress: pd.DataFrame) -> None:
    summary_rows = []
    for model_config in MODEL_CONFIGS:
        model_rows = _training_rows(
            progress[progress["short_name"] == model_config["short_name"]]
        )
        if model_rows.empty:
            continue
        summary_rows.append(
            {
                "short_name": model_config["short_name"],
                "model_id": model_config["model_id"],
                "model_root": model_rows["model_root"].iloc[0],
                "n_logged_training_rows": int(len(model_rows)),
                "max_global_epoch": float(model_rows["global_epoch"].max()),
                "min_loss": float(model_rows["loss"].min()),
                "final_loss": float(model_rows["loss"].dropna().iloc[-1]),
            }
        )
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "available_model_summary.csv", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress_frames = [
        progress
        for model_config in MODEL_CONFIGS
        if (progress := _load_model_progress(model_config)) is not None
    ]
    if not progress_frames:
        raise FileNotFoundError("No training_progress.csv files were found for the configured models.")

    progress = pd.concat(progress_frames, ignore_index=True)
    progress.to_csv(OUTPUT_DIR / "training_progress_with_global_axis.csv", index=False)

    for model_config in MODEL_CONFIGS:
        _write_model_training_dynamics(progress, model_config)
    _plot_available_models_summary(progress)

    print(f"Saved multi-model training dynamics plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
