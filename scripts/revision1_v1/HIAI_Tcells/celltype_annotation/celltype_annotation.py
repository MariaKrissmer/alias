from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Callable

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk


PROJECT_ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alias import evaluation as ev
from alias.evaluation.celltype_annotation import benchmark
from alias.evaluation.celltype_annotation.celltypist import (
    CellTypistAnnotationConfig,
    CellTypistTrainingConfig,
    run_celltypist_annotation,
    train_celltypist_model_from_dataset_dir,
)
from alias.evaluation.celltype_annotation.singler import (
    SingleRAnnotationConfig,
    run_singler_annotation,
)
from alias.util.load_hf_model import load_hf_dataset


DATASET_ID = "S2_heldout_donor_semantic_200k"
ANNOTATION_COLUMN = "AIFI_L2"
DATASET_DIR = PROJECT_ROOT / "out" / "data" / "revision1_v1" / "HIAI_Tcells" / DATASET_ID
ADATA_TEST_PATH = DATASET_DIR / "adata_test.h5ad"
SCRNA_TEST_PATH = DATASET_DIR / "datasets" / "scrna_test"
SCRNA_HF_DATASET = f"mariakrissmer/scrna_HIAI_Tcells_{DATASET_ID}"
OUTPUT_DIR = DATASET_DIR / "celltype_annotation"
FORCE_RERUN = os.environ.get("HIAI_TCELLS_ANNOTATION_FORCE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SOURCE_OBS_VALUE_MAP = {"AIFI_L2": {"gdT": "gamma delta T"}}


def _model_source(model_id: str) -> Path | str:
    repo_name = model_id.rsplit("/", maxsplit=1)[-1]
    for candidate in (
        PROJECT_ROOT / "models" / repo_name,
        PROJECT_ROOT / "models" / f"{repo_name}_all",
    ):
        if candidate.exists():
            return candidate
    return model_id


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


BENCHMARK_MODELS = [
    {
        "name": "CellTypist_HIAI_Tcells_S2_train_AIFI_L2",
        "label": "CellTypist",
        "kind": "celltypist",
        "model": OUTPUT_DIR / "models" / "CellTypist_HIAI_Tcells_S2_train_AIFI_L2.pkl",
        "model_source": "local",
        "train_from_dataset_dir": True,
        "check_expression": False,
        "reference_use_raw": True,
        "reference_obs_value_map": SOURCE_OBS_VALUE_MAP,
        "max_iter": 500,
        "n_jobs": 8,
    },
    {
        "name": "SingleR_HIAI_Tcells_S2_train_AIFI_L2",
        "label": "SingleR",
        "kind": "singler",
        "reference": "matching_train_split",
        "reference_use_raw": True,
        "reference_obs_value_map": SOURCE_OBS_VALUE_MAP,
        "max_reference_cells_per_label": 1000,
        "reference_sample_seed": 42,
        "num_threads": 4,
    },
]

COMPARISON_MODELS = [
    {
        "name": "Base",
        "kind": "sentence_transformer_similarity",
        "model": "neuml/pubmedbert-base-embeddings",
        "batch_size": 256,
        "max_cells": 20000,
    },
    {
        "name": "MI",
        "kind": "sentence_transformer_similarity",
        "model": _model_source("mariakrissmer/MI_HIAI_Tcells_N1_200k_lr5e5"),
        "batch_size": 256,
        "max_cells": 20000,
    },
    {
        "name": "MF",
        "kind": "sentence_transformer_similarity",
        "model": _model_source("mariakrissmer/MF_HIAI_Tcells_S3_200k_lr5e5"),
        "batch_size": 256,
        "max_cells": 20000,
    },
    {
        "name": "MG",
        "kind": "sentence_transformer_similarity",
        "model": _model_source("mariakrissmer/MG_HIAI_Tcells_S2_200k_lr5e5"),
        "batch_size": 256,
        "max_cells": 20000,
    },
    {
        "name": "MB",
        "kind": "sentence_transformer_similarity",
        "model": _model_source("mariakrissmer/MB_HIAI_Tcells_S2_N1_200k_lr5e5_e15"),
        "batch_size": 256,
        "max_cells": 20000,
    },
    {
        "name": "MJ",
        "kind": "sentence_transformer_similarity",
        "model": _model_source("mariakrissmer/MJ_HIAI_Tcells_S2_N3_200k_lr5e5"),
        "batch_size": 256,
        "max_cells": 20000,
    },
    {
        "name": "MH",
        "kind": "sentence_transformer_similarity",
        "model": _model_source("mariakrissmer/MH_HIAI_Tcells_S5_200k_lr5e5"),
        "batch_size": 256,
        "max_cells": 20000,
    },
    {
        "name": "ML",
        "kind": "sentence_transformer_similarity",
        "model": _model_source("mariakrissmer/ML_HIAI_Tcells_S7_N1_200k_lr5e5"),
        "batch_size": 256,
        "max_cells": 20000,
    },
]
MODEL_FILTER_ENV = "HIAI_TCELLS_ANNOTATION_MODELS"


def _parse_model_names(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [name.strip() for name in value.split(",") if name.strip()]
    return names or None


def _model_keys(model_config: dict) -> set[str]:
    return {
        str(model_config["name"]),
        str(model_config.get("label", model_config["name"])),
    }


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
        if _model_keys(model_config) & requested
    ]
    found = set().union(*[_model_keys(model_config) for model_config in selected]) if selected else set()
    missing = sorted(requested - found)
    if missing:
        print(f"Skipping unknown normal annotation model names: {missing}")
    if not selected:
        raise ValueError(f"No normal annotation models selected from: {model_names}")
    return selected


def _run_celltypist_model(model_config: dict) -> pd.DataFrame:
    model = model_config["model"]
    if model_config.get("train_from_dataset_dir"):
        model = train_celltypist_model_from_dataset_dir(
            CellTypistTrainingConfig(
                dataset_dir=DATASET_DIR,
                annotation_column=ANNOTATION_COLUMN,
                model_path=model,
                model_name=model_config["name"],
                reference_cache_dir=OUTPUT_DIR / "references",
                reference_use_raw=bool(model_config.get("reference_use_raw", False)),
                reference_obs_value_map=model_config.get("reference_obs_value_map"),
                force_rebuild_reference=bool(model_config.get("force_rebuild_reference", False)),
                force_retrain=bool(model_config.get("force_retrain", False)),
                check_expression=bool(model_config.get("check_expression", True)),
                max_iter=model_config.get("max_iter", 500),
                n_jobs=model_config.get("n_jobs", 8),
                details=(
                    f"CellTypist trained only on train cells from {DATASET_ID} "
                    f"using {ANNOTATION_COLUMN} labels."
                ),
                source=str(DATASET_DIR),
            )
        )
    return run_celltypist_annotation(
        CellTypistAnnotationConfig(
            adata_path=ADATA_TEST_PATH,
            annotation_column=ANNOTATION_COLUMN,
            model_name=model_config["name"],
            model=model,
            model_source=model_config.get("model_source", "local"),
            force_update=bool(model_config.get("force_update", False)),
        )
    )


def _run_similarity_model(model_config: dict) -> pd.DataFrame:
    model_output_dir = OUTPUT_DIR / model_config["name"]
    scrna_test = _load_scrna_test()
    evaluation_dict = {"scrna": {"test": scrna_test}}

    embedding_config = ev.GenEmbeddingsConfig(
        embedding_models=[str(model_config["model"])],
        model_type="sentence_transformer",
        batch_size=int(model_config.get("batch_size", 64)),
        output_dir=str(DATASET_DIR / "embeddings" / model_config["name"]),
        max_cells=int(model_config.get("max_cells", 5000)),
        annotation_column=ANNOTATION_COLUMN,
        force_regenerate=FORCE_RERUN,
    )

    embeddings_dict = ev.generate_embeddings(
        evaluation_dict=evaluation_dict,
        embedding_config=embedding_config,
    )

    similarity_config = ev.CellTypeSimilarityConfig(
        similarity_metric="cosine",
        bins=60,
        output_dir=model_output_dir / "label_similarity",
    )
    similarity_df = ev.cell_type_label_similarity(
        embeddings_dict=embeddings_dict,
        annotation_column=ANNOTATION_COLUMN,
        config=similarity_config,
    )
    similarity_df.to_csv(model_output_dir / "celltype_label_similarity.csv", index=False)

    return benchmark.make_similarity_predictions_from_embedding_artifacts(
        embeddings_dict=embeddings_dict,
        annotation_column=ANNOTATION_COLUMN,
    )


def _run_singler_model(model_config: dict) -> pd.DataFrame:
    return run_singler_annotation(
        SingleRAnnotationConfig(
            adata_path=ADATA_TEST_PATH,
            dataset_dir=DATASET_DIR,
            reference_cache_dir=OUTPUT_DIR / "references",
            annotation_column=ANNOTATION_COLUMN,
            model_name=model_config["name"],
            num_threads=int(model_config.get("num_threads", 1)),
            force_rebuild_reference=bool(model_config.get("force_rebuild_reference", False)),
            reference_use_raw=bool(model_config.get("reference_use_raw", False)),
            reference_obs_value_map=model_config.get("reference_obs_value_map"),
            max_reference_cells_per_label=model_config.get("max_reference_cells_per_label"),
            reference_sample_seed=int(model_config.get("reference_sample_seed", 42)),
        )
    )


def _prediction_runner(model_config: dict) -> Callable[[dict], pd.DataFrame]:
    if model_config["kind"] == "celltypist":
        return _run_celltypist_model
    if model_config["kind"] == "singler":
        return _run_singler_model
    if model_config["kind"] == "sentence_transformer_similarity":
        return _run_similarity_model
    raise ValueError(f"Unsupported annotation model kind: {model_config['kind']}")


def _evaluate_or_load_model(model_config: dict) -> pd.DataFrame:
    model_name = model_config["name"]
    model_dir = OUTPUT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_status = (
        "missing_predictions"
        if FORCE_RERUN
        else benchmark.get_annotation_cache_status(model_dir)
    )

    if cache_status == "complete":
        print(f"Using cached annotation benchmark for {model_name}: {model_dir}")
        return pd.read_csv(model_dir / "metrics_summary.csv")

    if cache_status == "missing_evaluation":
        print(f"Using cached predictions and regenerating evaluation for {model_name}")
        predictions = pd.read_csv(model_dir / "predictions.csv", index_col=0)
    else:
        print(f"Running annotation benchmark for {model_name}")
        predictions = _prediction_runner(model_config)(model_config)

    benchmark.write_annotation_evaluation(
        predictions=predictions,
        output_dir=model_dir,
        model_name=model_name,
        dataset_id=DATASET_ID,
        metadata={
            "dataset_dir": str(DATASET_DIR),
            "adata_test_path": str(ADATA_TEST_PATH),
            "scrna_test_path": str(SCRNA_TEST_PATH),
            "scrna_hf_dataset": SCRNA_HF_DATASET,
            "annotation_column": ANNOTATION_COLUMN,
            "model_kind": model_config["kind"],
            "model": str(model_config.get("model", model_config.get("reference", ""))),
            "model_source": str(model_config.get("model_source", "")),
            "max_reference_cells_per_label": model_config.get("max_reference_cells_per_label"),
            "reference_sample_seed": model_config.get("reference_sample_seed"),
        },
    )
    return pd.read_csv(model_dir / "metrics_summary.csv")


def _write_combined_outputs(summary_frames: list[pd.DataFrame]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(summary_frames, ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "model_comparison_metrics.csv", index=False)


def run(model_names: list[str] | None = None) -> pd.DataFrame:
    model_configs = _select_model_configs(
        [*BENCHMARK_MODELS, *COMPARISON_MODELS],
        model_names,
    )
    summary_frames = [_evaluate_or_load_model(model_config) for model_config in model_configs]
    _write_combined_outputs(summary_frames)
    return pd.concat(summary_frames, ignore_index=True)


def main() -> None:
    run(model_names=_parse_model_names(os.environ.get(MODEL_FILTER_ENV)))


if __name__ == "__main__":
    main()
