from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MPLCONFIG_DIR = PROJECT_ROOT / ".matplotlib"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import h5py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from alias.util.plots.color_definition import COLORMAPS


BASE_OUTPUT_DIR = PROJECT_ROOT / "out" / "data" / "revision1_v1" / "LaManno"
SOURCE_H5AD = BASE_OUTPUT_DIR / "lamanno_full.h5ad"
OUTPUT_DIR = BASE_OUTPUT_DIR / "heldout_sample_time_eval"
SAMPLE_COLUMN = "sample"
TIME_COLUMN = "time"

DATASETS = [
    ("S5", "S9_heldout_sample_semantic"),
    ("S6", "S10_heldout_timepoint_semantic"),
]
SAMPLE_REPORT_SPLITS = [("S5", "train"), ("S5", "test")]
TIME_REPORT_SPLITS = [("S6", "train"), ("S5", "test")]


def _decode_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_h5ad_obs_column(handle: h5py.File, column: str) -> list[Any]:
    obs = handle["obs"]
    if column not in obs:
        available = ", ".join(obs.keys())
        raise KeyError(f"Column '{column}' not found in obs. Available columns: {available}")

    obs_column = obs[column]
    if isinstance(obs_column, h5py.Group):
        categories = [_decode_value(value) for value in obs_column["categories"][:]]
        codes = obs_column["codes"][:]
        return [categories[code] if code >= 0 else None for code in codes]

    return [_decode_value(value) for value in obs_column[:]]


def read_obs_column_by_index(h5ad_path: Path, column: str) -> pd.Series:
    with h5py.File(h5ad_path, "r") as handle:
        obs = handle["obs"]
        index_column = _decode_value(obs.attrs.get("_index"))
        if index_column is None:
            raise KeyError(f"No obs index column declared in {h5ad_path}")

        obs_index = _read_h5ad_obs_column(handle, str(index_column))
        values = _read_h5ad_obs_column(handle, column)

    value_by_index = pd.Series(
        [None if value is None else str(value) for value in values],
        index=pd.Index(map(str, obs_index), name="cell_index"),
        name=column,
    )
    if value_by_index.index.has_duplicates:
        raise ValueError("AnnData obs index contains duplicates; cannot map split indices safely.")
    return value_by_index


def load_split_payloads(
    base_output_dir: Path,
    datasets: list[tuple[str, str]],
) -> list[tuple[str, dict[str, Any]]]:
    payloads = []
    for dataset_label, dataset_id in datasets:
        split_path = base_output_dir / dataset_id / "metadata" / "split_indices.json"
        if not split_path.exists():
            print(f"Warning: missing split indices for {dataset_label}: {split_path}")
            continue
        with split_path.open("r", encoding="utf-8") as handle:
            payloads.append((dataset_label, json.load(handle)))
    return payloads


def _values_for_indices(value_by_index: pd.Series, indices: list[str]) -> pd.Series:
    if not indices:
        return pd.Series(dtype=str)

    selected = value_by_index.reindex(map(str, indices))
    missing = selected[selected.isna()]
    if not missing.empty:
        missing_examples = ", ".join(map(str, missing.index[:5]))
        raise ValueError(
            f"{len(missing)} split indices were not found in the AnnData obs index. "
            f"Examples: {missing_examples}"
        )
    return selected.astype(str)


def build_presence_matrix(
    value_by_index: pd.Series,
    split_payloads: list[tuple[str, dict[str, Any]]],
    *,
    value_name: str,
    selected_splits: list[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = sorted(value_by_index.dropna().astype(str).unique())
    matrix_rows: dict[str, dict[str, int]] = {
        "full dataset": {value: 1 for value in values}
    }
    summary_rows = [
        {
            "row": "full dataset",
            "dataset": "full",
            "split": "full",
            "n_cells": int(value_by_index.notna().sum()),
            f"n_present_{value_name}": len(values),
            value_name: ";".join(values),
        }
    ]

    payload_by_label = dict(split_payloads)
    if selected_splits is None:
        split_plan = [
            (dataset_label, split_name)
            for dataset_label, _ in split_payloads
            for split_name in ["train", "test"]
        ]
    else:
        split_plan = selected_splits

    for dataset_label, split_name in split_plan:
        if dataset_label not in payload_by_label:
            print(f"Warning: missing split payload for {dataset_label}; skipping {dataset_label} {split_name}.")
            continue
        payload = payload_by_label[dataset_label]
        row_name = f"{dataset_label} {split_name}"
        indices = list(map(str, payload.get(f"{split_name}_indices", [])))
        split_values = sorted(_values_for_indices(value_by_index, indices).unique())
        present = set(split_values)
        matrix_rows[row_name] = {value: int(value in present) for value in values}
        summary_rows.append(
            {
                "row": row_name,
                "dataset": dataset_label,
                "split": split_name,
                "n_cells": len(indices),
                f"n_present_{value_name}": len(split_values),
                value_name: ";".join(split_values),
            }
        )

    matrix = pd.DataFrame.from_dict(matrix_rows, orient="index").loc[:, values]
    summary = pd.DataFrame(summary_rows)
    return matrix, summary


def plot_presence_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
    *,
    x_label: str,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_width = max(6, 0.28 * len(matrix.columns))
    fig_height = max(2.8, 0.5 * len(matrix.index))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        matrix,
        cmap=COLORMAPS["CM: Teal–White"],
        vmin=0,
        vmax=1,
        cbar=False,
        linewidths=0.25,
        linecolor="#d9d9d9",
        ax=ax,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Dataset split")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_column_report(
    *,
    value_by_index: pd.Series,
    split_payloads: list[tuple[str, dict[str, Any]]],
    value_name: str,
    output_stem: str,
    x_label: str,
    title: str,
    selected_splits: list[tuple[str, str]],
) -> tuple[Path, Path, Path]:
    matrix, summary = build_presence_matrix(
        value_by_index,
        split_payloads,
        value_name=value_name,
        selected_splits=selected_splits,
    )
    matrix_path = OUTPUT_DIR / f"{output_stem}_matrix.csv"
    summary_path = OUTPUT_DIR / f"{output_stem}_summary.csv"
    figure_stem = OUTPUT_DIR / output_stem
    matrix.to_csv(matrix_path)
    summary.to_csv(summary_path, index=False)
    plot_presence_heatmap(matrix, figure_stem, x_label=x_label, title=title)
    return matrix_path, summary_path, figure_stem.with_suffix(".pdf")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE_H5AD.exists():
        raise FileNotFoundError(f"Missing LaManno AnnData file: {SOURCE_H5AD}")

    split_payloads = load_split_payloads(BASE_OUTPUT_DIR, DATASETS)
    if not split_payloads:
        raise FileNotFoundError(
            f"No split_indices.json files found for configured datasets under {BASE_OUTPUT_DIR}"
        )

    sample_outputs = write_column_report(
        value_by_index=read_obs_column_by_index(SOURCE_H5AD, SAMPLE_COLUMN),
        split_payloads=split_payloads,
        value_name="sample",
        output_stem="sample_presence_heatmap",
        x_label="Sample",
        title="Sample presence across LaManno S5 train/test split",
        selected_splits=SAMPLE_REPORT_SPLITS,
    )
    time_outputs = write_column_report(
        value_by_index=read_obs_column_by_index(SOURCE_H5AD, TIME_COLUMN),
        split_payloads=split_payloads,
        value_name="time",
        output_stem="time_presence_heatmap",
        x_label="Time point",
        title="Time point presence across LaManno S6 train and S5 test splits",
        selected_splits=TIME_REPORT_SPLITS,
    )

    print("Saved LaManno sample report:")
    for output_path in sample_outputs:
        print(f"- {output_path}")
    print("Saved LaManno time report:")
    for output_path in time_outputs:
        print(f"- {output_path}")


if __name__ == "__main__":
    main()
