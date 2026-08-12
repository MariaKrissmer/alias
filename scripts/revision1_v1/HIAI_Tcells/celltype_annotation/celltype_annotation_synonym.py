from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "alias").exists():
            return candidate
    raise FileNotFoundError(f"Could not find project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from alias import evaluation as ev
from alias.evaluation.celltype_label_plots import EvaluationConfig
from alias.evaluation.celltype_annotation import benchmark
from alias.evaluation.embedding import clean_model_name, load_dataset_embedding_artifacts
from alias.util.similarity import evaluate_similarity
from alias.util.load_hf_model import load_hf_dataset

PLOTTING_SCRIPT_DIR = (
    PROJECT_ROOT / "scripts" / "revision1_v1" / "HIAI_Tcells" / "celltype_annotation"
)
if str(PLOTTING_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(PLOTTING_SCRIPT_DIR))

from _plotting_common import (  # noqa: E402
    ABLATION_BAR_X_SPACING,
    PAIRED_BAR_WIDTH,
    PLOT_HEIGHT,
    PUBLICATION_ABLATION_MODEL_ORDER,
    TICK_LABEL_SIZE,
    ModelSpec,
    PLOTS_DIR,
    load_metrics_summary,
    ordered_blue_model_palette,
    ordered_model_labels_by_metric,
    publication_model_labels,
    set_plot_style,
)


DATASET_ID = "S2_heldout_donor_semantic_200k"
ANNOTATION_COLUMN = "AIFI_L2"
DATASET_DIR = PROJECT_ROOT / "out" / "data" / "revision1_v1" / "HIAI_Tcells" / DATASET_ID
SCRNA_TEST_PATH = DATASET_DIR / "datasets" / "scrna_test"
SCRNA_HF_DATASET = f"mariakrissmer/scrna_HIAI_Tcells_{DATASET_ID}"
OUTPUT_DIR = DATASET_DIR / "celltype_annotation_synonym"
PLOT_OUTPUT_DIR = PLOTS_DIR / "synonym_annotation_effect"
SYNONYM_MAPPING_PATH = OUTPUT_DIR / "synonym_mapping.csv"
REQUIRED_SYNONYM_COLUMNS = {
    "canonical_label",
    "synonym_label",
    "is_canonical_label",
    "synonym_source",
}
FORCE_RERUN = os.environ.get("HIAI_TCELLS_SYNONYM_FORCE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
ABLATION_MODEL_ORDER = PUBLICATION_ABLATION_MODEL_ORDER
RUN_SYNONYM_LABEL_SIMILARITY_PLOTS = os.environ.get(
    "HIAI_TCELLS_SYNONYM_LABEL_PLOTS", "1"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RUN_EXAMPLE_SYNONYM_LABEL_SIMILARITY = RUN_SYNONYM_LABEL_SIMILARITY_PLOTS or os.environ.get(
    "HIAI_TCELLS_SYNONYM_EXAMPLE_PLOTS", "0"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RUN_UMAP_BEFORE_SYNONYM_LABEL_SIMILARITY = os.environ.get(
    "HIAI_TCELLS_SYNONYM_LABEL_UMAP", "0"
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RUN_UMAP_BEFORE_EXAMPLE_SYNONYM_LABEL_SIMILARITY = (
    RUN_UMAP_BEFORE_SYNONYM_LABEL_SIMILARITY
    or os.environ.get(
    "HIAI_TCELLS_SYNONYM_EXAMPLE_UMAP", "0"
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
)
SYNONYM_LABEL_SIMILARITY_MODEL_FILTER = {
    model.strip()
    for model in os.environ.get("HIAI_TCELLS_SYNONYM_LABEL_MODELS", "MB").split(",")
    if model.strip()
}
SYNONYM_LABEL_UMAP_MODEL_FILTER = {
    model.strip()
    for model in os.environ.get("HIAI_TCELLS_SYNONYM_LABEL_UMAP_MODELS", "MB").split(",")
    if model.strip()
}
EXAMPLE_SYNONYM_MODELS = ["MB"]
MAX_EXAMPLE_SYNONYM_LABELS = 2
EXAMPLE_SYNONYM_CANDIDATES = [
    ("Treg", "suppressor T cell"),
    ("MAIT", "mucosal-associated invariant T cell"),
    ("Naive CD8 T cell", "naive CD8-positive alpha-beta T cell"),
    ("Memory CD4 T cell", "CD4-positive, alpha-beta memory T cell"),
]
EXAMPLE_UMAP_CONFIG = EvaluationConfig(
    n_neighbors=50,
    min_dist=0.2,
    n_components=30,
    random_state=21,
)


def _model_source(model_id: str) -> Path | str:
    repo_name = model_id.rsplit("/", maxsplit=1)[-1]
    for candidate in (
        PROJECT_ROOT / "models" / repo_name,
        PROJECT_ROOT / "models" / f"{repo_name}_all",
    ):
        if candidate.exists():
            return candidate
    return model_id


COMPARISON_MODELS = [
    {
        "name": "Base",
        "kind": "sentence_transformer_similarity_synonyms",
        "model": "neuml/pubmedbert-base-embeddings",
        "batch_size": 256,
        "max_cells": 20000,
    },
    {
        "name": "MI",
        "kind": "sentence_transformer_similarity_synonyms",
        "model": _model_source("mariakrissmer/MI_HIAI_Tcells_N1_200k_lr5e5"),
        "batch_size": 256,
        "max_cells": 20000,
    },
    {
        "name": "MF",
        "kind": "sentence_transformer_similarity_synonyms",
        "model": _model_source("mariakrissmer/MF_HIAI_Tcells_S3_200k_lr5e5"),
        "batch_size": 256,
        "max_cells": 20000,
    },
    {
        "name": "MG",
        "kind": "sentence_transformer_similarity_synonyms",
        "model": _model_source("mariakrissmer/MG_HIAI_Tcells_S2_200k_lr5e5"),
        "batch_size": 256,
        "max_cells": 20000,
    },
    {
        "name": "MB",
        "kind": "sentence_transformer_similarity_synonyms",
        "model": _model_source("mariakrissmer/MB_HIAI_Tcells_S2_N1_200k_lr5e5_e15"),
        "batch_size": 256,
        "max_cells": 20000,
    },
    {
        "name": "MJ",
        "kind": "sentence_transformer_similarity_synonyms",
        "model": _model_source("mariakrissmer/MJ_HIAI_Tcells_S2_N3_200k_lr5e5"),
        "batch_size": 256,
        "max_cells": 20000,
    },
    {
        "name": "MH",
        "kind": "sentence_transformer_similarity_synonyms",
        "model": _model_source("mariakrissmer/MH_HIAI_Tcells_S5_200k_lr5e5"),
        "batch_size": 256,
        "max_cells": 20000,
    },
    {
        "name": "ML",
        "kind": "sentence_transformer_similarity_synonyms",
        "model": _model_source("mariakrissmer/ML_HIAI_Tcells_S7_N1_200k_lr5e5"),
        "batch_size": 256,
        "max_cells": 20000,
    },
]
MODEL_FILTER_ENV = "HIAI_TCELLS_SYNONYM_MODELS"


def _parse_model_names(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [name.strip() for name in value.split(",") if name.strip()]
    return names or None


def _select_model_configs(
    model_configs: list[dict],
    model_names: list[str] | None,
) -> list[dict]:
    if not model_names:
        return model_configs

    requested = set(model_names)
    selected = [
        model_config
        for model_config in model_configs
        if str(model_config["name"]) in requested
    ]
    found = {str(model_config["name"]) for model_config in selected}
    missing = sorted(requested - found)
    if missing:
        print(f"Skipping unknown synonym annotation model names: {missing}")
    if not selected:
        raise ValueError(f"No synonym annotation models selected from: {model_names}")
    return selected


def _load_scrna_test():
    if SCRNA_TEST_PATH.exists():
        try:
            print(f"Loading local scRNA test dataset from {SCRNA_TEST_PATH}")
            return load_from_disk(str(SCRNA_TEST_PATH))
        except Exception as error:
            print(
                f"load_from_disk failed for local scrna_test at {SCRNA_TEST_PATH}: "
                f"{type(error).__name__}: {error}"
            )
            arrow_files = sorted(SCRNA_TEST_PATH.glob("*.arrow"))
            if arrow_files:
                try:
                    import pyarrow as pa
                    import pyarrow.ipc as ipc

                    print(f"Loading local scRNA test dataset from Arrow files in {SCRNA_TEST_PATH}")
                    datasets = []
                    for path in arrow_files:
                        with pa.memory_map(str(path), "r") as source:
                            table = ipc.open_stream(source).read_all()
                        datasets.append(Dataset.from_pandas(table.to_pandas(), preserve_index=False))
                    return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
                except Exception as arrow_error:
                    raise RuntimeError(
                        f"Local scrna_test exists but could not be loaded from {SCRNA_TEST_PATH}. "
                        f"Current working directory: {Path.cwd()}. "
                        f"load_from_disk error: {type(error).__name__}: {error}. "
                        f"Arrow fallback error: {type(arrow_error).__name__}: {arrow_error}"
                    ) from arrow_error
            raise RuntimeError(
                f"Local scrna_test exists but could not be loaded from {SCRNA_TEST_PATH}. "
                f"Current working directory: {Path.cwd()}. "
                f"No Arrow files were found. "
                f"Original error: {type(error).__name__}: {error}"
            ) from error
    print(f"Local scrna_test not found at {SCRNA_TEST_PATH}")
    print(f"Falling back to Hugging Face dataset: {SCRNA_HF_DATASET}")

    dataset = load_hf_dataset(SCRNA_HF_DATASET)
    if isinstance(dataset, dict):
        for split_name in ("test", "scrna_test"):
            if split_name in dataset:
                print(f"Using Hugging Face split {split_name!r} from {SCRNA_HF_DATASET}")
                return dataset[split_name]
        raise KeyError(
            f"Could not find a test split in {SCRNA_HF_DATASET}. "
            f"Available splits: {sorted(dataset.keys())}"
        )
    return dataset


def _clean_synonym_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).replace("_", " ").split()).strip()


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.fillna(False).astype(str).str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "on"}
    )


def _load_synonym_map(observed_labels: set[str]) -> pd.DataFrame:
    if not SYNONYM_MAPPING_PATH.exists():
        raise FileNotFoundError(
            "No curated synonym mapping was found. Generate it once, then edit it "
            "manually if needed:\n"
            ".venv/bin/python "
            "scripts/revision1_v1/HIAI_Tcells/celltype_annotation/"
            "generate_synonym_mapping.py\n"
            f"Expected path: {SYNONYM_MAPPING_PATH}"
        )

    synonym_map = pd.read_csv(SYNONYM_MAPPING_PATH)
    missing_columns = sorted(REQUIRED_SYNONYM_COLUMNS - set(synonym_map.columns))
    if missing_columns:
        raise ValueError(
            f"{SYNONYM_MAPPING_PATH} is missing required columns: {missing_columns}"
        )

    synonym_map = synonym_map.copy()
    synonym_map["canonical_label"] = synonym_map["canonical_label"].map(_clean_synonym_text)
    synonym_map["synonym_label"] = synonym_map["synonym_label"].map(_clean_synonym_text)
    synonym_map["synonym_source"] = synonym_map["synonym_source"].map(_clean_synonym_text)
    synonym_map["synonym_source"] = synonym_map["synonym_source"].replace("", "manual")
    synonym_map["is_canonical_label"] = _coerce_bool_series(
        synonym_map["is_canonical_label"]
    )
    synonym_map = synonym_map[
        (synonym_map["canonical_label"] != "") & (synonym_map["synonym_label"] != "")
    ].copy()

    extra_labels = sorted(set(synonym_map["canonical_label"]) - observed_labels)
    if extra_labels:
        print(
            "Ignoring synonym rows for labels that are not present in this test set: "
            f"{extra_labels}"
        )
        synonym_map = synonym_map[
            synonym_map["canonical_label"].isin(observed_labels)
        ].copy()

    missing_labels = sorted(observed_labels - set(synonym_map["canonical_label"]))
    if missing_labels:
        raise ValueError(
            "The synonym mapping is missing observed labels. Add at least a canonical "
            f"row for each of these labels: {missing_labels}"
        )

    missing_canonical_rows = sorted(
        label
        for label in observed_labels
        if not (
            (synonym_map["canonical_label"] == label)
            & (synonym_map["synonym_label"] == label)
        ).any()
    )
    if missing_canonical_rows:
        raise ValueError(
            "The synonym mapping needs a canonical self-row for the original "
            f"annotation comparison: {missing_canonical_rows}"
        )

    dedupe_key = (
        synonym_map["canonical_label"].astype(str)
        + "\0"
        + synonym_map["synonym_label"].astype(str).str.casefold()
    )
    duplicated = dedupe_key.duplicated()
    if duplicated.any():
        print(
            "Dropping duplicate synonym rows after case-insensitive matching: "
            f"{int(duplicated.sum())}"
        )
        synonym_map = synonym_map.loc[~duplicated].copy()

    print(f"Loaded synonym mapping from {SYNONYM_MAPPING_PATH}")
    return synonym_map.reset_index(drop=True)


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


def _load_cell_and_synonym_embedding_artifacts(
    embeddings_dict: dict[str, dict[str, dict[str, Any]]],
    model_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    model_key = clean_model_name(str(model_config["model"]))
    dataset_meta = embeddings_dict[model_key]["scrna"]
    loaded = load_dataset_embedding_artifacts(
        dataset_meta,
        annotation_column=ANNOTATION_COLUMN,
    )
    artifacts = loaded["artifacts"]
    cell_artifact = artifacts["df_cells"]
    cell_umap = cell_artifact.get("umap")
    if cell_umap is None:
        cell_umap = _discover_cached_cell_umap(
            cell_artifact.get("metadata", {}),
            expected_n_cells=len(cell_artifact["dataframe"]),
        )
    return (
        cell_artifact["dataframe"],
        artifacts["df_additional"]["dataframe"],
        cell_umap,
    )


def _discover_cached_cell_umap(
    cell_metadata: dict[str, Any],
    *,
    expected_n_cells: int,
) -> pd.DataFrame | None:
    candidate_roots = []
    if cell_metadata.get("run_dir"):
        candidate_roots.append(Path(cell_metadata["run_dir"]))
    if cell_metadata.get("path"):
        candidate_roots.append(Path(cell_metadata["path"]).parent)

    seen_roots: set[Path] = set()
    candidates: list[Path] = []
    for root in candidate_roots:
        root = root.resolve()
        if root in seen_roots or not root.exists():
            continue
        seen_roots.add(root)
        candidates.extend(root.glob("umap/**/df_cells_umap.parquet"))

    for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
        cell_umap = pd.read_parquet(path)
        if {"UMAP1", "UMAP2"}.issubset(cell_umap.columns) and len(cell_umap) == expected_n_cells:
            print(f"Using cached UMAP coordinates from {path}")
            return cell_umap
        print(
            "Skipping cached UMAP coordinates with incompatible shape/columns: "
            f"{path}"
        )
    return None


def _load_cell_and_synonym_embeddings(
    embeddings_dict: dict[str, dict[str, dict[str, Any]]],
    model_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_embeddings, synonym_embeddings, _ = _load_cell_and_synonym_embedding_artifacts(
        embeddings_dict,
        model_config,
    )
    return (
        cell_embeddings,
        synonym_embeddings,
    )


def _generate_or_reuse_synonym_embeddings(
    model_config: dict,
    scrna_test,
    synonym_map: pd.DataFrame,
) -> dict[str, dict[str, dict[str, Any]]]:
    synonym_labels = synonym_map["synonym_label"].tolist()
    evaluation_dict = {"scrna": {"test": scrna_test}}

    embedding_config = ev.GenEmbeddingsConfig(
        embedding_models=[str(model_config["model"])],
        model_type="sentence_transformer",
        batch_size=int(model_config.get("batch_size", 64)),
        output_dir=str(DATASET_DIR / "embeddings"),
        max_cells=int(model_config.get("max_cells", 5000)),
        annotation_column=ANNOTATION_COLUMN,
        additional_data=synonym_labels,
        force_regenerate=FORCE_RERUN,
    )
    return ev.generate_embeddings(
        evaluation_dict=evaluation_dict,
        embedding_config=embedding_config,
    )


def _embedding_lookup_by_label_text(synonym_embeddings: pd.DataFrame) -> pd.DataFrame:
    if "data" not in synonym_embeddings.columns:
        raise ValueError("Synonym embeddings are missing the synonym text column 'data'.")

    numeric_columns = synonym_embeddings.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_columns:
        raise ValueError("No numeric synonym embedding columns found.")
    return synonym_embeddings.groupby("data", sort=False)[numeric_columns].mean()


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^\w\d\-\.]+", "_", str(value)).strip("_")
    return safe[:120] or "label"


def _select_example_synonyms(synonym_map: pd.DataFrame) -> pd.DataFrame:
    available = {
        (row["canonical_label"], row["synonym_label"]): row
        for _, row in synonym_map.iterrows()
    }
    selected_rows = []
    for canonical_label, synonym_label in EXAMPLE_SYNONYM_CANDIDATES:
        row = available.get((canonical_label, synonym_label))
        if row is None:
            print(
                "Skipping example synonym plot because the mapping is missing: "
                f"{canonical_label!r} -> {synonym_label!r}"
            )
            continue
        selected_rows.append(dict(row))
        if len(selected_rows) >= MAX_EXAMPLE_SYNONYM_LABELS:
            break

    if not selected_rows:
        print("No configured example synonym labels were found in the synonym mapping.")
        return pd.DataFrame(columns=synonym_map.columns)
    return pd.DataFrame(selected_rows)


def _select_synonym_label_similarity_rows(synonym_map: pd.DataFrame) -> pd.DataFrame:
    if RUN_SYNONYM_LABEL_SIMILARITY_PLOTS:
        return synonym_map.sort_values(
            ["canonical_label", "is_canonical_label", "synonym_label"],
            ascending=[True, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
    return _select_example_synonyms(synonym_map)


def _should_plot_synonym_label_similarity_for_model(model_name: str) -> bool:
    if RUN_SYNONYM_LABEL_SIMILARITY_PLOTS:
        return (
            not SYNONYM_LABEL_SIMILARITY_MODEL_FILTER
            or model_name in SYNONYM_LABEL_SIMILARITY_MODEL_FILTER
        )
    return model_name in EXAMPLE_SYNONYM_MODELS


def _should_run_umap_before_synonym_label_similarity(model_name: str) -> bool:
    if RUN_UMAP_BEFORE_EXAMPLE_SYNONYM_LABEL_SIMILARITY:
        return (
            not SYNONYM_LABEL_UMAP_MODEL_FILTER
            or model_name in SYNONYM_LABEL_UMAP_MODEL_FILTER
        )
    return (
        RUN_SYNONYM_LABEL_SIMILARITY_PLOTS
        and model_name in SYNONYM_LABEL_UMAP_MODEL_FILTER
    )


def _write_example_synonym_label_similarity(
    model_config: dict,
    scrna_test,
    synonym_map: pd.DataFrame,
) -> None:
    if not RUN_EXAMPLE_SYNONYM_LABEL_SIMILARITY:
        return
    model_name = str(model_config["name"])
    if not _should_plot_synonym_label_similarity_for_model(model_name):
        return

    label_rows = _select_synonym_label_similarity_rows(synonym_map)
    if label_rows.empty:
        return

    embeddings_dict = _generate_or_reuse_synonym_embeddings(
        model_config=model_config,
        scrna_test=scrna_test,
        synonym_map=synonym_map,
    )
    if _should_run_umap_before_synonym_label_similarity(model_name):
        embeddings_dict = ev.umap_plots(
            embeddings_dict=embeddings_dict,
            annotation_column=ANNOTATION_COLUMN,
            output_dir=str(PLOT_OUTPUT_DIR / "example_synonym_umap_context"),
            evaluation_config=EXAMPLE_UMAP_CONFIG,
        )
    cell_embeddings, synonym_embeddings, cell_umap = _load_cell_and_synonym_embedding_artifacts(
        embeddings_dict,
        model_config=model_config,
    )
    label_embedding_lookup = _embedding_lookup_by_label_text(synonym_embeddings)
    missing = sorted(
        set(label_rows["synonym_label"].astype(str)) - set(label_embedding_lookup.index)
    )
    if missing:
        raise ValueError(f"Missing embeddings for synonym labels: {missing}")

    cell_matrix = _numeric_embedding_values(
        cell_embeddings,
        excluded_columns={ANNOTATION_COLUMN},
    )
    if cell_umap is None:
        print(
            "No cached UMAP coordinates found for synonym label similarity plots. "
            "Writing ROC-AUC and histogram plots only. Run ev.umap_plots first "
            "if you want UMAP overlays."
        )
    cell_annotations = cell_embeddings[ANNOTATION_COLUMN].astype(str)
    output_name = (
        "synonym_label_similarity"
        if RUN_SYNONYM_LABEL_SIMILARITY_PLOTS
        else "example_synonym_label_similarity"
    )
    plot_root = PLOT_OUTPUT_DIR / output_name / model_name
    plot_root.mkdir(parents=True, exist_ok=True)
    all_results = []

    for row in label_rows.itertuples(index=False):
        canonical_label = str(row.canonical_label)
        synonym_label = str(row.synonym_label)
        output_dir = (
            plot_root
            / f"{_safe_filename(canonical_label)}__{_safe_filename(synonym_label)}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        other_embedding = label_embedding_lookup.loc[[synonym_label]].to_numpy(dtype=np.float32)
        ground_truth = pd.DataFrame(
            {canonical_label: cell_annotations == canonical_label},
            index=cell_embeddings.index,
        )
        plot_label = (
            canonical_label
            if bool(row.is_canonical_label)
            else f"{canonical_label}: {synonym_label}"
        )
        results_df, _ = evaluate_similarity(
            cell_embeddings=cell_matrix,
            other_embeddings=other_embedding,
            other_labels=[plot_label],
            ground_truth=ground_truth,
            cell_umap=cell_umap,
            other_umap=None,
            similarity_metric="cosine",
            output_dir=output_dir,
            bins=60,
        )
        results_df["model_name"] = model_name
        results_df["canonical_label"] = canonical_label
        results_df["synonym_label"] = synonym_label
        results_df["is_canonical_label"] = bool(row.is_canonical_label)
        results_df["synonym_source"] = str(row.synonym_source)
        results_df["output_dir"] = str(output_dir)
        results_df.to_csv(output_dir / "results_df.csv", index=False)
        all_results.append(results_df)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(plot_root / f"{output_name}_results.csv", index=False)
        metadata = {
            "model_name": model_name,
            "dataset_id": DATASET_ID,
            "annotation_column": ANNOTATION_COLUMN,
            "n_cells": int(len(cell_embeddings)),
            "n_annotation_labels": int(len(label_rows)),
            "annotation_labels": label_rows[
                ["canonical_label", "synonym_label", "synonym_source"]
            ].to_dict(orient="records"),
            "ran_umap_before_example_similarity": (
                _should_run_umap_before_synonym_label_similarity(model_name)
            ),
            "synonym_label_umap_model_filter": sorted(SYNONYM_LABEL_UMAP_MODEL_FILTER),
            "cell_umap_available": cell_umap is not None,
            "cell_umap_note": (
                "Reused cached cell UMAP coordinates from embedding artifacts."
                if cell_umap is not None
                else "No UMAP overlays were produced. Run ev.umap_plots before this script."
            ),
        }
        with (plot_root / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)


def _candidate_label_variants(synonym_map: pd.DataFrame) -> list[dict[str, Any]]:
    canonical_labels = sorted(synonym_map["canonical_label"].unique())
    variants = [
        {
            "variant_type": "Original annotation",
            "variant_index": 0,
            "labels": synonym_map[
                synonym_map["is_canonical_label"]
            ][["canonical_label", "synonym_label"]].rename(
                columns={"synonym_label": "label_text"}
            ),
        }
    ]

    noncanonical = (
        synonym_map[~synonym_map["is_canonical_label"]]
        .groupby("canonical_label", sort=True)["synonym_label"]
        .apply(list)
        .to_dict()
    )
    labels_without_synonyms = sorted(set(canonical_labels) - set(noncanonical))
    if labels_without_synonyms:
        raise ValueError(
            "Cannot build synonym variants because these labels only have their "
            f"canonical label: {labels_without_synonyms}"
        )
    max_synonyms = max(len(values) for values in noncanonical.values())
    for variant_index in range(max_synonyms):
        rows = []
        for canonical_label in canonical_labels:
            synonyms = noncanonical[canonical_label]
            rows.append(
                {
                    "canonical_label": canonical_label,
                    "label_text": synonyms[variant_index % len(synonyms)],
                }
            )
        variants.append(
            {
                "variant_type": "Synonym annotation",
                "variant_index": variant_index + 1,
                "labels": pd.DataFrame(rows),
            }
        )
    return variants


def _make_predictions_from_candidate_labels(
    cell_embeddings: pd.DataFrame,
    label_embedding_lookup: pd.DataFrame,
    candidate_labels: pd.DataFrame,
) -> pd.DataFrame:
    if ANNOTATION_COLUMN not in cell_embeddings.columns:
        raise ValueError(f"Cell embeddings are missing {ANNOTATION_COLUMN!r}.")

    missing_labels = sorted(set(candidate_labels["label_text"]) - set(label_embedding_lookup.index))
    if missing_labels:
        raise ValueError(f"Missing embeddings for candidate labels: {missing_labels}")

    cell_matrix = _numeric_embedding_values(
        cell_embeddings,
        excluded_columns={ANNOTATION_COLUMN},
    )
    label_matrix = label_embedding_lookup.loc[candidate_labels["label_text"]].to_numpy(
        dtype=np.float32
    )
    similarity = _cosine_similarity(cell_matrix, label_matrix)
    top_indices = np.argmax(similarity, axis=1)

    return pd.DataFrame(
        {
            "true_label": cell_embeddings[ANNOTATION_COLUMN].astype(str).to_numpy(),
            "predicted_label": candidate_labels["canonical_label"].to_numpy()[top_indices],
            "predicted_annotation_label": candidate_labels["label_text"].to_numpy()[top_indices],
            "score": similarity[np.arange(similarity.shape[0]), top_indices].astype(float),
        },
        index=cell_embeddings.index,
    )


def _with_variant_metadata(
    df: pd.DataFrame,
    *,
    model_name: str,
    variant_type: str,
    variant_index: int,
) -> pd.DataFrame:
    annotated = df.copy()
    annotated.insert(0, "variant_index", variant_index)
    annotated.insert(0, "variant_type", variant_type)
    annotated.insert(0, "model_name", model_name)
    return annotated


def _label_source_lookup(synonym_map: pd.DataFrame) -> dict[tuple[str, str], str]:
    return {
        (str(row.canonical_label), str(row.synonym_label)): str(row.synonym_source)
        for row in synonym_map.itertuples(index=False)
    }


def _annotate_candidate_labels(
    candidate_labels: pd.DataFrame,
    *,
    synonym_map: pd.DataFrame,
    model_name: str,
    variant_type: str,
    variant_index: int,
) -> pd.DataFrame:
    source_lookup = _label_source_lookup(synonym_map)
    label_df = candidate_labels.copy()
    label_df["synonym_source"] = [
        source_lookup.get((str(row.canonical_label), str(row.label_text)), "unknown")
        for row in label_df.itertuples(index=False)
    ]
    return _with_variant_metadata(
        label_df,
        model_name=model_name,
        variant_type=variant_type,
        variant_index=variant_index,
    )


def _metrics_for_candidate_labels(
    *,
    model_name: str,
    variant_type: str,
    variant_index: int,
    cell_embeddings: pd.DataFrame,
    label_embedding_lookup: pd.DataFrame,
    candidate_labels: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    predictions = _make_predictions_from_candidate_labels(
        cell_embeddings=cell_embeddings,
        label_embedding_lookup=label_embedding_lookup,
        candidate_labels=candidate_labels,
    )
    metrics = benchmark.compute_annotation_metrics(
        predictions,
        model_name=model_name,
        dataset_id=DATASET_ID,
    )
    summary = {
        "model_name": model_name,
        "variant_type": variant_type,
        "variant_index": variant_index,
        "accuracy": metrics["summary"]["accuracy"],
        "balanced_accuracy": metrics["summary"]["balanced_accuracy"],
        "macro_f1": metrics["summary"]["macro_f1"],
        "weighted_f1": metrics["summary"]["weighted_f1"],
        "n_cells": metrics["summary"]["n_cells"],
        "n_true_labels": metrics["summary"]["n_true_labels"],
        "n_predicted_labels": metrics["summary"]["n_predicted_labels"],
    }
    per_celltype = _with_variant_metadata(
        metrics["per_celltype"],
        model_name=model_name,
        variant_type=variant_type,
        variant_index=variant_index,
    )
    return summary, per_celltype


def _metric_variant_tables_for_model(
    model_config: dict,
    scrna_test,
    synonym_map: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    embeddings_dict = _generate_or_reuse_synonym_embeddings(
        model_config=model_config,
        scrna_test=scrna_test,
        synonym_map=synonym_map,
    )
    cell_embeddings, synonym_embeddings = _load_cell_and_synonym_embeddings(
        embeddings_dict,
        model_config=model_config,
    )
    label_embedding_lookup = _embedding_lookup_by_label_text(synonym_embeddings)

    summary_rows = []
    per_celltype_rows = []
    label_rows = []
    for variant in _candidate_label_variants(synonym_map):
        summary, per_celltype = _metrics_for_candidate_labels(
            model_name=model_config["name"],
            variant_type=variant["variant_type"],
            variant_index=variant["variant_index"],
            cell_embeddings=cell_embeddings,
            label_embedding_lookup=label_embedding_lookup,
            candidate_labels=variant["labels"],
        )
        summary_rows.append(summary)
        per_celltype_rows.append(per_celltype)
        label_rows.append(
            _annotate_candidate_labels(
                variant["labels"],
                synonym_map=synonym_map,
                model_name=model_config["name"],
                variant_type=variant["variant_type"],
                variant_index=variant["variant_index"],
            )
        )
    return {
        "summary": pd.DataFrame(summary_rows),
        "per_celltype": pd.concat(per_celltype_rows, ignore_index=True),
        "label_map": pd.concat(label_rows, ignore_index=True),
    }


def _single_synonym_metric_tables_for_model(
    model_config: dict,
    scrna_test,
    synonym_map: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    embeddings_dict = _generate_or_reuse_synonym_embeddings(
        model_config=model_config,
        scrna_test=scrna_test,
        synonym_map=synonym_map,
    )
    cell_embeddings, synonym_embeddings = _load_cell_and_synonym_embeddings(
        embeddings_dict,
        model_config=model_config,
    )
    label_embedding_lookup = _embedding_lookup_by_label_text(synonym_embeddings)
    canonical_labels = sorted(synonym_map["canonical_label"].unique())
    base_labels = pd.DataFrame(
        {
            "canonical_label": canonical_labels,
            "label_text": canonical_labels,
        }
    )

    summary_rows = []
    per_celltype_rows = []
    for synonym_index, row in enumerate(synonym_map.itertuples(index=False)):
        candidate_labels = base_labels.copy()
        candidate_labels.loc[
            candidate_labels["canonical_label"] == row.canonical_label,
            "label_text",
        ] = row.synonym_label
        variant_type = (
            "Original annotation"
            if bool(row.is_canonical_label)
            else "Single synonym annotation"
        )
        summary, per_celltype = _metrics_for_candidate_labels(
            model_name=model_config["name"],
            variant_type=variant_type,
            variant_index=synonym_index,
            cell_embeddings=cell_embeddings,
            label_embedding_lookup=label_embedding_lookup,
            candidate_labels=candidate_labels,
        )
        synonym_metadata = {
            "canonical_label": str(row.canonical_label),
            "synonym_label": str(row.synonym_label),
            "is_canonical_label": bool(row.is_canonical_label),
            "synonym_source": str(row.synonym_source),
        }
        summary_rows.append({**synonym_metadata, **summary})
        for key, value in synonym_metadata.items():
            per_celltype[key] = value
        per_celltype_rows.append(per_celltype)

    return {
        "summary": pd.DataFrame(summary_rows),
        "per_celltype": pd.concat(per_celltype_rows, ignore_index=True),
    }


def _normal_annotation_metrics_by_model(model_names: list[str]) -> dict[str, pd.Series]:
    metrics_by_model = {}
    for model_name in model_names:
        metrics = load_metrics_summary(ModelSpec(model_name, model_name))
        if metrics is None or metrics.empty:
            continue
        metrics_by_model[model_name] = metrics.iloc[0]
    return metrics_by_model


def _metric_comparison_rows(
    metric_variant_df: pd.DataFrame,
    *,
    normal_annotation_metrics: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    normal_annotation_metrics = normal_annotation_metrics or {}
    rows = []
    for model_name, model_df in metric_variant_df.groupby("model_name", sort=False):
        original = model_df[model_df["variant_type"] == "Original annotation"]
        synonyms = model_df[model_df["variant_type"] == "Synonym annotation"]
        if original.empty or synonyms.empty:
            continue
        normal_metrics = normal_annotation_metrics.get(str(model_name))

        for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
            original_value = (
                normal_metrics[metric]
                if normal_metrics is not None and metric in normal_metrics
                else original.iloc[0][metric]
            )
            rows.append(
                {
                    "model_name": model_name,
                    "metric": metric,
                    "annotation_type": "Original annotation",
                    "mean_value": float(original_value),
                    "sd_value": 0.0,
                    "n_variants": 1,
                }
            )
            rows.append(
                {
                    "model_name": model_name,
                    "metric": metric,
                    "annotation_type": "Synonym annotation",
                    "mean_value": float(synonyms[metric].mean()),
                    "sd_value": float(synonyms[metric].std(ddof=1)),
                    "n_variants": int(len(synonyms)),
                }
            )
    return pd.DataFrame(rows)


def _true_annotation_balanced_accuracy_order(
    *,
    model_order: list[str],
    fallback_order_df: pd.DataFrame,
) -> list[str]:
    metrics_frames = [
        load_metrics_summary(ModelSpec(label, label))
        for label in model_order
    ]
    metrics_frames = [
        frame for frame in metrics_frames if frame is not None and not frame.empty
    ]
    if metrics_frames:
        metrics_df = pd.concat(metrics_frames, ignore_index=True)
        return ordered_model_labels_by_metric(
            metrics_df,
            model_labels=model_order,
            metric="balanced_accuracy",
            model_column="plot_model",
            ascending=False,
        )
    return ordered_model_labels_by_metric(
        fallback_order_df,
        model_labels=model_order,
        metric="balanced_accuracy",
        model_column="plot_model",
        ascending=False,
    )


def _plot_balanced_accuracy_synonym_ablation(comparison_df: pd.DataFrame) -> None:
    metric_df = comparison_df[comparison_df["metric"] == "balanced_accuracy"].copy()
    if metric_df.empty:
        return

    original_df = metric_df[
        metric_df["annotation_type"] == "Original annotation"
    ][["model_name", "mean_value"]].rename(columns={"mean_value": "original"})
    synonym_df = metric_df[
        metric_df["annotation_type"] == "Synonym annotation"
    ][["model_name", "mean_value", "sd_value"]].rename(
        columns={"mean_value": "synonym_mean", "sd_value": "synonym_sd"}
    )
    plot_df = (
        original_df.merge(synonym_df, on="model_name", how="inner")
        .reset_index(drop=True)
    )
    if plot_df.empty:
        return

    configured_order = [
        label for label in ABLATION_MODEL_ORDER if label in set(plot_df["model_name"])
    ]
    configured_order.extend(
        label
        for label in plot_df["model_name"].astype(str).tolist()
        if label not in configured_order
    )
    model_order = configured_order
    plot_df["model_name"] = pd.Categorical(
        plot_df["model_name"],
        categories=model_order,
        ordered=True,
    )
    plot_df = plot_df.sort_values("model_name", kind="mergesort").reset_index(drop=True)
    palette = ordered_blue_model_palette(model_order)
    x_positions = np.arange(len(model_order)) * ABLATION_BAR_X_SPACING
    width = PAIRED_BAR_WIDTH

    fig, ax = plt.subplots(figsize=(3.25, PLOT_HEIGHT))
    ax.bar(
        x_positions - width / 2,
        plot_df["original"].astype(float),
        width=width,
        color=[palette[model] for model in model_order],
        edgecolor="0.15",
        linewidth=0.35,
        label="Original annotation",
    )
    ax.bar(
        x_positions + width / 2,
        plot_df["synonym_mean"].astype(float),
        width=width,
        yerr=plot_df["synonym_sd"].astype(float),
        capsize=2,
        color=[palette[model] for model in model_order],
        edgecolor="0.15",
        linewidth=0.35,
        hatch="////",
        error_kw={"elinewidth": 0.6, "ecolor": "0.2", "capthick": 0.6},
        label="Synonym annotation",
    )
    for patch in ax.patches[len(model_order):]:
        patch._hatch_color = (1, 1, 1, 1)

    ax.set_xlabel("")
    ax.set_ylabel("Balanced Accuracy", fontsize=9.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        publication_model_labels(model_order),
        rotation=45,
        ha="right",
        fontsize=TICK_LABEL_SIZE,
    )
    ax.tick_params(axis="x", which="major", bottom=True, length=3, width=0.7, color="0.2")
    ax.tick_params(axis="y", labelsize=8.5)
    if len(x_positions):
        ax.set_xlim(min(x_positions) - 0.36, max(x_positions) + 0.36)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    sns.despine(ax=ax)
    for output_dir, stem in [
        (PLOT_OUTPUT_DIR, "balanced_accuracy_original_vs_synonym_ablation"),
        (OUTPUT_DIR, "balanced_accuracy_original_vs_synonym_annotations"),
    ]:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight", dpi=300)
        fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def _make_synonym_predictions(
    cell_embeddings: pd.DataFrame,
    synonym_embeddings: pd.DataFrame,
    synonym_map: pd.DataFrame,
) -> pd.DataFrame:
    if ANNOTATION_COLUMN not in cell_embeddings.columns:
        raise ValueError(f"Cell embeddings are missing {ANNOTATION_COLUMN!r}.")
    if "data" not in synonym_embeddings.columns:
        raise ValueError("Synonym embeddings are missing the synonym text column 'data'.")

    synonym_lookup = synonym_map.set_index("synonym_label")["canonical_label"].to_dict()
    synonym_labels = synonym_embeddings["data"].astype(str).tolist()
    missing_synonyms = sorted(set(synonym_labels) - set(synonym_lookup))
    if missing_synonyms:
        raise ValueError(f"Missing synonym mapping for embedded labels: {missing_synonyms}")

    cell_matrix = _numeric_embedding_values(
        cell_embeddings,
        excluded_columns={ANNOTATION_COLUMN},
    )
    synonym_matrix = _numeric_embedding_values(
        synonym_embeddings,
        excluded_columns={"data"},
    )
    similarity = _cosine_similarity(cell_matrix, synonym_matrix)
    top_indices = np.argmax(similarity, axis=1)
    true_labels = cell_embeddings[ANNOTATION_COLUMN].astype(str).to_numpy()

    synonym_to_indices: dict[str, list[int]] = {}
    canonical_to_indices: dict[str, list[int]] = {}
    for synonym_index, synonym_label in enumerate(synonym_labels):
        canonical_label = synonym_lookup[synonym_label]
        synonym_to_indices.setdefault(synonym_label, []).append(synonym_index)
        canonical_to_indices.setdefault(canonical_label, []).append(synonym_index)

    true_label_scores = []
    true_label_synonym_mean_scores = []
    true_label_synonym_max_scores = []
    for row_index, true_label in enumerate(true_labels):
        canonical_indices = canonical_to_indices[true_label]
        exact_indices = synonym_to_indices.get(true_label, canonical_indices)
        true_label_scores.append(float(similarity[row_index, exact_indices].mean()))
        true_label_synonym_mean_scores.append(float(similarity[row_index, canonical_indices].mean()))
        true_label_synonym_max_scores.append(float(similarity[row_index, canonical_indices].max()))

    predicted_synonyms = np.array(synonym_labels, dtype=object)[top_indices]
    predicted_labels = np.array([synonym_lookup[synonym] for synonym in predicted_synonyms])

    return pd.DataFrame(
        {
            "true_label": true_labels,
            "predicted_label": predicted_labels,
            "predicted_synonym_label": predicted_synonyms,
            "score": similarity[np.arange(similarity.shape[0]), top_indices].astype(float),
            "true_label_score": true_label_scores,
            "true_label_synonym_mean_score": true_label_synonym_mean_scores,
            "true_label_synonym_max_score": true_label_synonym_max_scores,
        },
        index=cell_embeddings.index,
    )


def _write_synonym_score_summary(predictions: pd.DataFrame, output_dir: Path) -> None:
    summary = {
        "n_cells": int(len(predictions)),
        "mean_predicted_synonym_score": float(predictions["score"].mean()),
        "mean_true_label_score": float(predictions["true_label_score"].mean()),
        "mean_true_label_synonym_mean_score": float(
            predictions["true_label_synonym_mean_score"].mean()
        ),
        "mean_true_label_synonym_max_score": float(
            predictions["true_label_synonym_max_score"].mean()
        ),
    }
    pd.DataFrame([summary]).to_csv(output_dir / "synonym_score_summary.csv", index=False)
    with (output_dir / "synonym_score_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def _run_synonym_similarity_model(
    model_config: dict,
    scrna_test,
    synonym_map: pd.DataFrame,
) -> pd.DataFrame:
    embeddings_dict = _generate_or_reuse_synonym_embeddings(
        model_config=model_config,
        scrna_test=scrna_test,
        synonym_map=synonym_map,
    )
    cell_embeddings, synonym_embeddings = _load_cell_and_synonym_embeddings(
        embeddings_dict,
        model_config=model_config,
    )
    return _make_synonym_predictions(
        cell_embeddings=cell_embeddings,
        synonym_embeddings=synonym_embeddings,
        synonym_map=synonym_map,
    )


def _evaluate_or_load_model(
    model_config: dict,
    scrna_test,
    synonym_map: pd.DataFrame,
) -> pd.DataFrame:
    model_name = model_config["name"]
    model_dir = OUTPUT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_status = (
        "missing_predictions"
        if FORCE_RERUN
        else benchmark.get_annotation_cache_status(model_dir)
    )

    if cache_status == "complete":
        print(f"Using cached synonym annotation benchmark for {model_name}: {model_dir}")
        return pd.read_csv(model_dir / "metrics_summary.csv")

    if cache_status == "missing_evaluation":
        print(f"Using cached synonym predictions and regenerating evaluation for {model_name}")
        predictions = pd.read_csv(model_dir / "predictions.csv", index_col=0)
    else:
        print(f"Running synonym annotation benchmark for {model_name}")
        predictions = _run_synonym_similarity_model(
            model_config=model_config,
            scrna_test=scrna_test,
            synonym_map=synonym_map,
        )

    benchmark.write_annotation_evaluation(
        predictions=predictions,
        output_dir=model_dir,
        model_name=model_name,
        dataset_id=DATASET_ID,
        metadata={
            "dataset_dir": str(DATASET_DIR),
            "scrna_test_path": str(SCRNA_TEST_PATH),
            "scrna_hf_dataset": SCRNA_HF_DATASET,
            "annotation_column": ANNOTATION_COLUMN,
            "model_kind": model_config["kind"],
            "model": str(model_config["model"]),
            "n_synonym_labels": int(len(synonym_map)),
            "n_canonical_labels": int(synonym_map["canonical_label"].nunique()),
        },
    )
    _write_synonym_score_summary(predictions, model_dir)
    return pd.read_csv(model_dir / "metrics_summary.csv")


def _write_synonym_metric_comparison(
    model_configs: list[dict],
    scrna_test,
    synonym_map: pd.DataFrame,
) -> None:
    variant_tables = [
        _metric_variant_tables_for_model(
            model_config=model_config,
            scrna_test=scrna_test,
            synonym_map=synonym_map,
        )
        for model_config in model_configs
    ]
    single_synonym_tables = [
        _single_synonym_metric_tables_for_model(
            model_config=model_config,
            scrna_test=scrna_test,
            synonym_map=synonym_map,
        )
        for model_config in model_configs
    ]
    variant_df = pd.concat(
        [tables["summary"] for tables in variant_tables],
        ignore_index=True,
    )
    variant_per_celltype_df = pd.concat(
        [tables["per_celltype"] for tables in variant_tables],
        ignore_index=True,
    )
    variant_label_map_df = pd.concat(
        [tables["label_map"] for tables in variant_tables],
        ignore_index=True,
    )
    single_synonym_df = pd.concat(
        [tables["summary"] for tables in single_synonym_tables],
        ignore_index=True,
    )
    single_synonym_per_celltype_df = pd.concat(
        [tables["per_celltype"] for tables in single_synonym_tables],
        ignore_index=True,
    )
    model_names = sorted(variant_df["model_name"].astype(str).unique())
    normal_annotation_metrics = _normal_annotation_metrics_by_model(model_names)
    comparison_df = _metric_comparison_rows(
        variant_df,
        normal_annotation_metrics=normal_annotation_metrics,
    )
    variant_df.to_csv(OUTPUT_DIR / "synonym_metric_variant_scores.csv", index=False)
    variant_per_celltype_df.to_csv(
        OUTPUT_DIR / "synonym_metric_variant_per_celltype_scores.csv",
        index=False,
    )
    variant_label_map_df.to_csv(
        OUTPUT_DIR / "synonym_metric_variant_label_map.csv",
        index=False,
    )
    single_synonym_df.to_csv(
        OUTPUT_DIR / "single_synonym_annotation_metrics.csv",
        index=False,
    )
    single_synonym_per_celltype_df.to_csv(
        OUTPUT_DIR / "single_synonym_annotation_per_celltype_metrics.csv",
        index=False,
    )
    comparison_df.to_csv(OUTPUT_DIR / "synonym_metric_comparison.csv", index=False)
    for model_name in sorted(single_synonym_df["model_name"].unique()):
        model_dir = OUTPUT_DIR / str(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)
        single_synonym_df[
            single_synonym_df["model_name"] == model_name
        ].to_csv(model_dir / "single_synonym_annotation_metrics.csv", index=False)
        single_synonym_per_celltype_df[
            single_synonym_per_celltype_df["model_name"] == model_name
        ].to_csv(
            model_dir / "single_synonym_annotation_per_celltype_metrics.csv",
            index=False,
        )
        variant_df[variant_df["model_name"] == model_name].to_csv(
            model_dir / "synonym_metric_variant_scores.csv",
            index=False,
        )
        variant_per_celltype_df[
            variant_per_celltype_df["model_name"] == model_name
        ].to_csv(
            model_dir / "synonym_metric_variant_per_celltype_scores.csv",
            index=False,
        )
        variant_label_map_df[
            variant_label_map_df["model_name"] == model_name
        ].to_csv(model_dir / "synonym_metric_variant_label_map.csv", index=False)
    PLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_balanced_accuracy_synonym_ablation(comparison_df)


def _write_combined_outputs(summary_frames: list[pd.DataFrame]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(summary_frames, ignore_index=True)
    combined_path = OUTPUT_DIR / "model_comparison_metrics.csv"
    combined.to_csv(combined_path, index=False)


def run(model_names: list[str] | None = None) -> pd.DataFrame:
    set_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scrna_test = _load_scrna_test()
    observed_labels = set(map(str, scrna_test[ANNOTATION_COLUMN]))

    synonym_map = _load_synonym_map(observed_labels)
    model_configs = _select_model_configs(COMPARISON_MODELS, model_names)

    summary_frames = [
        _evaluate_or_load_model(
            model_config=model_config,
            scrna_test=scrna_test,
            synonym_map=synonym_map,
        )
        for model_config in model_configs
    ]
    _write_combined_outputs(summary_frames)
    _write_synonym_metric_comparison(
        model_configs=model_configs,
        scrna_test=scrna_test,
        synonym_map=synonym_map,
    )
    for model_config in model_configs:
        _write_example_synonym_label_similarity(
            model_config=model_config,
            scrna_test=scrna_test,
            synonym_map=synonym_map,
        )
    return pd.concat(summary_frames, ignore_index=True)


def main() -> None:
    run(model_names=_parse_model_names(os.environ.get(MODEL_FILTER_ENV)))


if __name__ == "__main__":
    main()
