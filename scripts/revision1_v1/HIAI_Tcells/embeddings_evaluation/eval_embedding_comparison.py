from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

ANNOTATION_PLOT_DIR = PROJECT_ROOT / "scripts" / "revision1_v1" / "HIAI_Tcells" / "celltype_annotation"
if str(ANNOTATION_PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(ANNOTATION_PLOT_DIR))

import anndata as ad
from datasets import Dataset, load_from_disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.ipc as pa_ipc
import seaborn as sns

from alias import evaluation as ev
from alias.evaluation.embedding_evaluation import (
    align_embedding_frame_to_adata,
    compute_expression_pca_embeddings,
    compute_gene_list_pca_embeddings,
    load_external_embedding_frame,
)
from alias.evaluation.embedding import find_reusable_embedding_run, resolve_artifact_root
from alias.util.append_adata import add_embeddings_to_adata
from alias.util.load_hf_model import load_hf_dataset
from alias.util.plots.color_definition import (
    PUBLICATION_ABLATION_MODEL_PALETTE,
    SCIB_BENCHMARK_COLORS,
    SCIB_SCORE_COLORS,
    SCIB_TABLE_HEADER_COLOR,
)
from _plotting_common import (  # noqa: E402
    ModelSpec as AnnotationModelSpec,
    PUBLICATION_ABLATION_MODEL_ORDER,
    concat_available as concat_available_annotation_frames,
    load_metrics_summary as load_annotation_metrics_summary,
    ordered_blue_model_palette as annotation_ordered_blue_model_palette,
    ordered_model_labels_by_metric,
    publication_model_label,
    publication_model_labels,
)


DATASET_ID = "S2_heldout_donor_semantic_200k"
ANNOTATION_COLUMN = "AIFI_L2"
SCRNA_HF_DATASET = f"mariakrissmer/scrna_HIAI_Tcells_{DATASET_ID}"

REVISION_DATA_ROOT = Path(
    os.environ.get("ALIAS_REVISION_DATA_ROOT", PROJECT_ROOT / "out" / "data" / "revision1_v1")
)
DATASET_DIR = REVISION_DATA_ROOT / "HIAI_Tcells" / DATASET_ID
EMBEDDING_ROOT = Path(
    os.environ.get("HIAI_TCELLS_EMBEDDING_ROOT", DATASET_DIR / "embeddings")
)
ANNOTATION_DIR = Path(
    os.environ.get("HIAI_TCELLS_ANNOTATION_OUTPUT_DIR", DATASET_DIR / "celltype_annotation")
)
ADATA_TEST_PATH = DATASET_DIR / "adata_test.h5ad"
SCRNA_TEST_PATH = DATASET_DIR / "datasets" / "scrna_test"
OUTPUT_DIR = Path(
    os.environ.get(
        "HIAI_TCELLS_SCIB_OUTPUT_DIR",
        DATASET_DIR / "embeddings_evaluation" / "scib_embedding_comparison",
    )
)
EXTERNAL_EMBEDDING_DIR = DATASET_DIR / "embeddings_evaluation" / "external_embeddings"
EMBEDDING_MAPPING_PATH = OUTPUT_DIR / "embedding_obsm_keys.csv"
SCIB_RAW_RESULTS_PATH = OUTPUT_DIR / "scib_results_raw.csv"
SCIB_SCALED_RESULTS_PATH = OUTPUT_DIR / "scib_results_minmax_scaled.csv"
SCIB_AGGREGATE_SCORES_PATH = OUTPUT_DIR / "scib_aggregate_scores.csv"
SCIB_ABLATION_SCORES_PATH = OUTPUT_DIR / "scib_ablation_scores.csv"
SCIB_BENCHMARK_SCORES_PATH = OUTPUT_DIR / "scib_benchmark_scores.csv"

BATCH_KEY = os.environ.get("HIAI_TCELLS_SCIB_BATCH_KEY", "batch")
LABEL_KEY = os.environ.get("HIAI_TCELLS_SCIB_LABEL_KEY", "cell_type")
N_JOBS = int(os.environ.get("HIAI_TCELLS_SCIB_N_JOBS", "6"))
BATCH_SIZE = int(os.environ.get("HIAI_TCELLS_SCIB_BATCH_SIZE", "256"))
MAX_CELLS = int(os.environ.get("HIAI_TCELLS_SCIB_MAX_CELLS", "20000"))
PCA_N_COMPONENTS = int(os.environ.get("HIAI_TCELLS_SCIB_PCA_N_COMPONENTS", "50"))
RANDOM_STATE = int(os.environ.get("HIAI_TCELLS_SCIB_RANDOM_STATE", "42"))


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


FORCE_ALL = _env_bool("HIAI_TCELLS_SCIB_FORCE")
FORCE_SCIB_SCORES = FORCE_ALL or _env_bool("HIAI_TCELLS_SCIB_FORCE_SCORES")
FORCE_EMBEDDINGS = FORCE_ALL or _env_bool("HIAI_TCELLS_SCIB_FORCE_EMBEDDINGS")
REQUIRE_CACHED_EMBEDDINGS = _env_bool("HIAI_TCELLS_SCIB_REQUIRE_CACHED_EMBEDDINGS")
SAVE_ADATA = os.environ.get("HIAI_TCELLS_SCIB_SAVE_ADATA", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MODEL_FILTER_ENV = "HIAI_TCELLS_SCIB_MODELS"

AGGREGATE_SCORE_COLUMNS = ["Batch correction", "Bio conservation", "Total"]
ABLATION_ORDER = PUBLICATION_ABLATION_MODEL_ORDER
ANNOTATION_COLOR_ORDER = [*PUBLICATION_ABLATION_MODEL_ORDER, "ML"]
ANNOTATION_MODEL_SPECS = [
    AnnotationModelSpec(
        "MB",
        "MB",
        ("MB_HIAI_Tcells_S2_N1_200k_lr5e5_e15", "MB_HIAI_Tcells_S2_N1_200k_lr5e5_e15_epoch_15_ncbi"),
    ),
    AnnotationModelSpec("MJ", "MJ", ("MJ_HIAI_Tcells_S2_N3_200k_lr5e5",)),
    AnnotationModelSpec("MG", "MG", ("MG_HIAI_Tcells_S2_200k_lr5e5", "MG_HIAI_Tcells_S2_200k")),
    AnnotationModelSpec("MF", "MF", ("MF_HIAI_Tcells_S3_200k_lr5e5",)),
    AnnotationModelSpec("MH", "MH", ("MH_HIAI_Tcells_S5_200k_lr5e5",)),
    AnnotationModelSpec("MI", "MI", ("MI_HIAI_Tcells_N1_200k_lr5e5",)),
    AnnotationModelSpec("Base", "PubMedBERTBase"),
    AnnotationModelSpec("ML", "ML", ("ML_HIAI_Tcells_S7_N1_200k_lr5e5",)),
]
BENCHMARK_ORDER = [
    "PCA_expression",
    "PCA_rank_weighted",
    "PCA_one_hot",
    "scVI_census_pretrained",
    "scVI_census_dataset_adapted",
    "scGPT",
    "MG",
    "MB",
]
SCIB_TABLE_DISPLAY_LABELS = {
    "PCA_expression": "PCA_expression",
    "PCA_rank_weighted": "PCA_rank",
    "PCA_one_hot": "PCA_onehot",
    "scVI_census_pretrained": "scVI_pretrain",
    "scVI_census_dataset_adapted": "scVI_adapted",
    "scGPT": "scGPT",
}
SCORE_COLOR_MAP = SCIB_SCORE_COLORS
ANNOTATION_REFERENCE_FIGSIZE = (3.0, 1.75)
SCIB_MODEL_X_SPACING = 0.43
SCIB_SCORE_OFFSETS = {
    "Batch correction": -0.09,
    "Bio conservation": 0.0,
    "Total": 0.09,
}
BENCHMARK_BAR_X_SPACING = 0.46
BENCHMARK_BAR_WIDTH = 0.20


@dataclass(frozen=True)
class EmbeddingSpec:
    name: str
    kind: str
    model: str | Path | None = None
    weighting: str | None = None
    external_path: Path | None = None
    include_ablation: bool = False
    include_benchmark: bool = False


def print_dependency_guidance() -> None:
    print("Dependency guidance for .venv_annotation:", flush=True)
    print(
        '  uv pip install --python .venv_annotation/bin/python "scib-metrics>=0.5.0"',
        flush=True,
    )
    print(
        '  uv pip install --python .venv_annotation/bin/python "scikit-learn>=1.4" "pyarrow>=15"',
        flush=True,
    )
    print(
        "Optional only if future scripts generate external embeddings instead of loading files:",
        flush=True,
    )
    print(
        '  uv pip install --python .venv_annotation/bin/python "scvi-tools>=1.2"',
        flush=True,
    )
    print('  uv pip install --python .venv_annotation/bin/python "scgpt"', flush=True)


def _external_path(
    name: str,
    env_name: str,
    *,
    fallback_env_names: tuple[str, ...] = (),
) -> Path:
    configured = os.environ.get(env_name)
    if configured is None:
        configured = next(
            (
                os.environ[fallback_env_name]
                for fallback_env_name in fallback_env_names
                if os.environ.get(fallback_env_name)
            ),
            None,
        )
    if configured:
        return Path(configured)
    if name == "scVI_census_pretrained":
        return EXTERNAL_EMBEDDING_DIR / "scVI_census_pretrained.parquet"
    if name == "scVI_census_dataset_adapted":
        return EXTERNAL_EMBEDDING_DIR / "scVI_census_dataset_adapted.parquet"
    return EXTERNAL_EMBEDDING_DIR / f"{name}.parquet"


EMBEDDING_SPECS = [
    EmbeddingSpec(
        name="Base",
        kind="sentence_transformer",
        model="neuml/pubmedbert-base-embeddings",
        include_ablation=True,
    ),
    EmbeddingSpec(
        name="MI",
        kind="sentence_transformer",
        model="mariakrissmer/ALIAS-MD",
        include_ablation=True,
    ),
    EmbeddingSpec(
        name="MF",
        kind="sentence_transformer",
        model="mariakrissmer/ALIAS-MC",
        include_ablation=True,
    ),
    EmbeddingSpec(
        name="MG",
        kind="sentence_transformer",
        model="mariakrissmer/ALIAS-MB",
        include_ablation=True,
        include_benchmark=True,
    ),
    EmbeddingSpec(
        name="MB",
        kind="sentence_transformer",
        model="mariakrissmer/ALIAS-MA",
        include_ablation=True,
        include_benchmark=True,
    ),
    EmbeddingSpec(
        name="MH",
        kind="sentence_transformer",
        model="mariakrissmer/ALIAS-MC-star",
        include_ablation=True,
    ),
    EmbeddingSpec(
        name="MJ",
        kind="sentence_transformer",
        model="mariakrissmer/ALIAS-MA-star",
        include_ablation=True,
    ),
    EmbeddingSpec(
        name="PCA_expression",
        kind="expression_pca",
        include_benchmark=True,
    ),
    EmbeddingSpec(
        name="PCA_rank_weighted",
        kind="gene_list_pca",
        weighting="inverse_sqrt_rank",
        include_benchmark=True,
    ),
    EmbeddingSpec(
        name="PCA_one_hot",
        kind="gene_list_pca",
        weighting="one_hot",
        include_benchmark=True,
    ),
    EmbeddingSpec(
        name="scVI_census_pretrained",
        kind="external",
        external_path=_external_path(
            "scVI_census_pretrained",
            "HIAI_TCELLS_SCIB_SCVI_CENSUS_PRETRAINED_PATH",
            fallback_env_names=("HIAI_TCELLS_SCIB_SCVI_CENSUS_PATH",),
        ),
        include_benchmark=True,
    ),
    EmbeddingSpec(
        name="scVI_census_dataset_adapted",
        kind="external",
        external_path=_external_path(
            "scVI_census_dataset_adapted",
            "HIAI_TCELLS_SCIB_SCVI_CENSUS_DATASET_ADAPTED_PATH",
        ),
        include_benchmark=True,
    ),
    EmbeddingSpec(
        name="scGPT",
        kind="external",
        external_path=_external_path("scGPT", "HIAI_TCELLS_SCIB_SCGPT_PATH"),
        include_benchmark=True,
    ),
]


def _selected_embedding_specs() -> list[EmbeddingSpec]:
    configured = os.environ.get(MODEL_FILTER_ENV)
    if not configured or configured.strip().lower() == "all":
        return EMBEDDING_SPECS

    requested = [name.strip() for name in configured.split(",") if name.strip()]
    requested_set = set(requested)
    selected = [spec for spec in EMBEDDING_SPECS if spec.name in requested_set]
    missing = sorted(requested_set - {spec.name for spec in selected})
    if missing:
        print(f"Skipping unknown embedding evaluation model names: {missing}")
    if not selected:
        raise ValueError(f"No embedding evaluation models selected from {requested}")
    return selected


def _normalize_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "on"})


def _requested_model_names(specs: list[EmbeddingSpec]) -> set[str]:
    return {spec.name for spec in specs}


def _load_cached_scib_scores_if_complete(specs: list[EmbeddingSpec]) -> pd.DataFrame | None:
    if FORCE_SCIB_SCORES:
        print(
            "Ignoring cached scIB scores because score recomputation is enabled "
            "(HIAI_TCELLS_SCIB_FORCE_SCORES or HIAI_TCELLS_SCIB_FORCE)."
        )
        return None

    if not SCIB_AGGREGATE_SCORES_PATH.exists():
        return None

    aggregate_scores = pd.read_csv(SCIB_AGGREGATE_SCORES_PATH)
    required_columns = {
        "model_name",
        "include_ablation",
        "include_benchmark",
        *AGGREGATE_SCORE_COLUMNS,
    }
    missing_columns = required_columns - set(aggregate_scores.columns)
    if missing_columns:
        print(
            "Ignoring cached scIB scores because "
            f"{SCIB_AGGREGATE_SCORES_PATH.name} is missing columns: "
            f"{sorted(missing_columns)}"
        )
        return None

    requested_names = _requested_model_names(specs)
    cached_names = set(aggregate_scores["model_name"].astype(str))
    missing_models = sorted(requested_names - cached_names)
    if missing_models:
        print(f"Cached scIB scores are incomplete; missing models: {missing_models}")
        return None

    if EMBEDDING_MAPPING_PATH.exists():
        mapping = pd.read_csv(EMBEDDING_MAPPING_PATH, keep_default_na=False)
        if not {"model_name", "model", "external_path", "kind"}.issubset(mapping.columns):
            print(
                "Ignoring cached scIB scores because "
                f"{EMBEDDING_MAPPING_PATH.name} lacks model source columns."
            )
            return None
        mapping_names = set(mapping["model_name"].astype(str))
        missing_mapping_models = sorted(requested_names - mapping_names)
        if missing_mapping_models:
            print(
                "Cached scIB embedding mapping is incomplete; missing models: "
                f"{missing_mapping_models}"
            )
            return None
        mapping = mapping.set_index("model_name")
        for spec in specs:
            saved = mapping.loc[spec.name]
            if isinstance(saved, pd.DataFrame):
                print(f"Ignoring cached scIB scores because {spec.name} has duplicate mappings.")
                return None
            if (
                str(saved["kind"]) != spec.kind
                or str(saved["model"]) != str(spec.model or "")
                or str(saved["external_path"]) != str(spec.external_path or "")
            ):
                print(f"Ignoring cached scIB scores because {spec.name} source changed.")
                return None
    else:
        print(
            "Ignoring cached scIB scores because "
            f"{EMBEDDING_MAPPING_PATH.name} is missing."
        )
        return None

    score_df = aggregate_scores[
        aggregate_scores["model_name"].astype(str).isin(requested_names)
    ].copy()
    score_df["include_ablation"] = _normalize_bool_series(score_df["include_ablation"])
    score_df["include_benchmark"] = _normalize_bool_series(score_df["include_benchmark"])

    for column in AGGREGATE_SCORE_COLUMNS:
        score_df[column] = pd.to_numeric(score_df[column], errors="coerce")
    if score_df[AGGREGATE_SCORE_COLUMNS].isna().any().any():
        print(
            "Ignoring cached scIB scores because at least one aggregate score is not numeric."
        )
        return None

    return score_df.reset_index(drop=True)


def _write_score_outputs(
    aggregate_scores: pd.DataFrame,
    *,
    write_aggregate: bool = True,
) -> None:
    if write_aggregate:
        aggregate_scores.to_csv(SCIB_AGGREGATE_SCORES_PATH, index=False)
    aggregate_scores[aggregate_scores["include_ablation"]].to_csv(
        SCIB_ABLATION_SCORES_PATH,
        index=False,
    )
    aggregate_scores[aggregate_scores["include_benchmark"]].to_csv(
        SCIB_BENCHMARK_SCORES_PATH,
        index=False,
    )


def _write_cached_run_metadata(
    *,
    specs: list[EmbeddingSpec],
    aggregate_scores: pd.DataFrame,
) -> None:
    metadata = {
        "dataset_id": DATASET_ID,
        "dataset_dir": str(DATASET_DIR),
        "selected_models": [spec.name for spec in specs],
        "reused_cached_scib_scores": True,
        "cache_source": str(SCIB_AGGREGATE_SCORES_PATH),
        "n_cached_models": int(aggregate_scores["model_name"].nunique()),
        "force_all": FORCE_ALL,
        "force_scib_scores": FORCE_SCIB_SCORES,
        "force_embeddings": FORCE_EMBEDDINGS,
        "model_filter_env": MODEL_FILTER_ENV,
    }
    with (OUTPUT_DIR / "scib_cache_reuse_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _load_arrow_dataset_frame(path: Path) -> pd.DataFrame:
    arrow_files = sorted(path.glob("data-*.arrow"))
    if not arrow_files:
        raise FileNotFoundError(f"No Arrow data files found in {path}")

    frames = []
    for arrow_file in arrow_files:
        with pa_ipc.open_stream(str(arrow_file)) as reader:
            frames.append(reader.read_all().to_pandas())
    return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]


def _load_scrna_test() -> pd.DataFrame:
    if SCRNA_TEST_PATH.exists():
        try:
            return pd.DataFrame(load_from_disk(str(SCRNA_TEST_PATH)))
        except Exception as error:
            print(
                f"Could not load local scrna_test from {SCRNA_TEST_PATH}: "
                f"{type(error).__name__}: {error}"
            )
            print("Reading local scrna_test Arrow files directly instead.")
            return _load_arrow_dataset_frame(SCRNA_TEST_PATH)

    dataset = load_hf_dataset(SCRNA_HF_DATASET)
    if isinstance(dataset, dict):
        for split_name in ("test", "scrna_test"):
            if split_name in dataset:
                return pd.DataFrame(dataset[split_name])
        raise KeyError(
            f"Could not find a test split in {SCRNA_HF_DATASET}. "
            f"Available splits: {sorted(dataset.keys())}. "
            "This benchmark needs cell-indexed test rows aligned to adata_test.h5ad; "
            "scrna_train/scrna_eval triplet splits are not valid substitutes."
        )
    return pd.DataFrame(dataset)


def _load_adata_test() -> ad.AnnData:
    if not ADATA_TEST_PATH.exists():
        raise FileNotFoundError(f"adata_test.h5ad not found: {ADATA_TEST_PATH}")
    return ad.read_h5ad(ADATA_TEST_PATH)


def _select_scrna_rows(scrna_test) -> pd.DataFrame:
    df = pd.DataFrame(scrna_test)
    if MAX_CELLS and len(df) > MAX_CELLS:
        df = df.sample(n=MAX_CELLS, random_state=RANDOM_STATE)
    if "index" not in df.columns:
        df = df.copy()
        df["index"] = df.index.astype(str)
    df["index"] = df["index"].astype(str)
    if "gene_list" in df.columns:
        df["gene_list"] = df["gene_list"].map(
            lambda genes: genes.tolist() if hasattr(genes, "tolist") else list(genes)
        )
    return df.reset_index(drop=True)


def _subset_adata_to_rows(adata: ad.AnnData, scrna_rows: pd.DataFrame) -> ad.AnnData:
    cell_ids = scrna_rows["index"].astype(str).tolist()
    missing = pd.Index(cell_ids).difference(adata.obs_names.astype(str))
    if len(missing):
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"adata_test.h5ad is missing {len(missing)} selected scRNA cells. "
            f"First missing ids: {preview}"
        )
    return adata[cell_ids].copy()


def _ensure_scib_obs_columns(adata: ad.AnnData) -> tuple[ad.AnnData, str]:
    if ANNOTATION_COLUMN not in adata.obs:
        raise KeyError(f"Missing annotation column in adata.obs: {ANNOTATION_COLUMN}")

    adata.obs[LABEL_KEY] = adata.obs[ANNOTATION_COLUMN].astype(str).astype("category")

    source_batch_key = BATCH_KEY
    if BATCH_KEY not in adata.obs:
        batch_candidates = [
            "batch_id",
            "sample.sampleKitGuid",
            "subject.subjectGuid",
            "pool_id",
            "chip_id",
        ]
        source_batch_key = next(
            (candidate for candidate in batch_candidates if candidate in adata.obs),
            None,
        )
        if source_batch_key is None:
            raise KeyError(
                "Could not create scIB batch column. None of these columns exist in "
                f"adata.obs: {batch_candidates}"
            )
        adata.obs[BATCH_KEY] = adata.obs[source_batch_key].astype(str)

    adata.obs[BATCH_KEY] = adata.obs[BATCH_KEY].astype(str).fillna("unknown").astype("category")
    print(f"Using adata.obs[{source_batch_key!r}] as scIB batch column {BATCH_KEY!r}")
    return adata, source_batch_key


def _append_sentence_transformer_embedding(
    adata: ad.AnnData,
    scrna_rows: pd.DataFrame,
    spec: EmbeddingSpec,
) -> tuple[ad.AnnData, Path]:
    embedding_config = ev.GenEmbeddingsConfig(
        embedding_models=[str(spec.model)],
        model_type="sentence_transformer",
        batch_size=BATCH_SIZE,
        output_dir=str(EMBEDDING_ROOT / spec.name),
        max_cells=MAX_CELLS,
        annotation_column=ANNOTATION_COLUMN,
        force_regenerate=FORCE_EMBEDDINGS,
    )
    if FORCE_EMBEDDINGS and REQUIRE_CACHED_EMBEDDINGS:
        raise ValueError("Cannot force embeddings in cache-only scIB mode.")

    annotation_metadata_path = ANNOTATION_DIR / spec.name / "metadata.json"
    annotation_source_matches = False
    if annotation_metadata_path.exists():
        with annotation_metadata_path.open("r", encoding="utf-8") as handle:
            annotation_source_matches = json.load(handle).get("model") == str(spec.model)

    if annotation_source_matches and not FORCE_EMBEDDINGS:
        embeddings_root = resolve_artifact_root(embedding_config.output_dir, "embeddings")
        cached_run = find_reusable_embedding_run(
            embeddings_root=embeddings_root,
            dataset_name="scrna",
            model_name=str(spec.model),
            embedding_config=embedding_config,
        )
        if cached_run is not None:
            with (cached_run / "metadata.json").open("r", encoding="utf-8") as handle:
                cached_metadata = json.load(handle)
            cell_path = Path(cached_metadata["artifacts"]["df_cells"]["path"])
            cached_cells = pd.read_parquet(cell_path)
            if cached_cells.index.has_duplicates:
                raise ValueError(f"Cached embeddings have duplicate cell IDs: {cell_path}")
            if len(cached_cells) != adata.n_obs:
                raise ValueError(
                    f"Cached embeddings have {len(cached_cells)} cells, expected {adata.n_obs}: {cell_path}"
                )
            updated = align_embedding_frame_to_adata(
                adata, cached_cells, obsm_key=spec.name, copy=False, require_all_cells=True
            )
            print(f"Reusing {spec.name} cell embeddings from {cell_path}")
            return updated, cell_path

    if REQUIRE_CACHED_EMBEDDINGS:
        raise FileNotFoundError(
            f"No matching cached {spec.name} embeddings for {spec.model} in "
            f"{embedding_config.output_dir}. Check {annotation_metadata_path}."
        )

    scrna_test = Dataset.from_pandas(scrna_rows, preserve_index=False)
    evaluation_dict = {"scrna": {"test": scrna_test}}
    embeddings_dict = ev.generate_embeddings(
        evaluation_dict=evaluation_dict,
        embedding_config=embedding_config,
    )
    model_key = next(iter(embeddings_dict))
    updated = add_embeddings_to_adata(
        adata,
        embeddings_dict,
        model_key=model_key,
        dataset_key="scrna",
        annotation_column=ANNOTATION_COLUMN,
        obsm_key=spec.name,
        copy=False,
        require_all_cells=True,
    )
    cell_path = Path(embeddings_dict[model_key]["scrna"]["df_cells"]["path"])
    return updated, cell_path


def _append_gene_list_pca_embedding(
    adata: ad.AnnData,
    scrna_rows: pd.DataFrame,
    spec: EmbeddingSpec,
) -> ad.AnnData:
    embeddings = compute_gene_list_pca_embeddings(
        scrna_rows,
        weighting=spec.weighting,
        n_components=PCA_N_COMPONENTS,
        random_state=RANDOM_STATE,
    )
    return align_embedding_frame_to_adata(
        adata,
        embeddings,
        obsm_key=spec.name,
        copy=False,
        require_all_cells=True,
    )


def _append_expression_pca_embedding(adata: ad.AnnData, spec: EmbeddingSpec) -> ad.AnnData:
    embeddings = compute_expression_pca_embeddings(
        adata,
        n_components=PCA_N_COMPONENTS,
        random_state=RANDOM_STATE,
    )
    return align_embedding_frame_to_adata(
        adata,
        embeddings,
        obsm_key=spec.name,
        copy=False,
        require_all_cells=True,
    )


def _append_external_embedding(adata: ad.AnnData, spec: EmbeddingSpec) -> ad.AnnData | None:
    if spec.external_path is None:
        raise ValueError(f"External embedding spec {spec.name} does not define a path.")
    path = spec.external_path
    if not path.exists() and path.suffix == ".parquet":
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            path = csv_path
    if not path.exists():
        print(
            f"Skipping {spec.name}: no precomputed embedding file found at "
            f"{spec.external_path} or {spec.external_path.with_suffix('.csv')}"
        )
        return None

    embeddings = load_external_embedding_frame(path)
    return align_embedding_frame_to_adata(
        adata,
        embeddings,
        obsm_key=spec.name,
        copy=False,
        require_all_cells=True,
    )


def _append_embedding(
    adata: ad.AnnData,
    scrna_rows: pd.DataFrame,
    spec: EmbeddingSpec,
) -> tuple[ad.AnnData, dict | None]:
    print(f"Preparing embedding for {spec.name} ({spec.kind})")
    embedding_source_path = ""
    if spec.kind == "sentence_transformer":
        updated, cell_path = _append_sentence_transformer_embedding(adata, scrna_rows, spec)
        embedding_source_path = str(cell_path)
    elif spec.kind == "gene_list_pca":
        updated = _append_gene_list_pca_embedding(adata, scrna_rows, spec)
    elif spec.kind == "expression_pca":
        updated = _append_expression_pca_embedding(adata, spec)
    elif spec.kind == "external":
        updated = _append_external_embedding(adata, spec)
        if updated is None:
            return adata, None
    else:
        raise ValueError(f"Unsupported embedding kind: {spec.kind}")

    return updated, {
        "model_name": spec.name,
        "kind": spec.kind,
        "model": str(spec.model or ""),
        "weighting": spec.weighting or "",
        "external_path": str(spec.external_path or ""),
        "embedding_source_path": embedding_source_path,
        "embedding_timestamp": Path(embedding_source_path).parent.name if embedding_source_path else "",
        "obsm_key": spec.name,
        "include_ablation": spec.include_ablation,
        "include_benchmark": spec.include_benchmark,
        "embedding_dim": int(updated.obsm[spec.name].shape[1]),
    }


def _ordered_palette(model_names: list[str]) -> dict[str, tuple[float, float, float, float]]:
    return {
        model: PUBLICATION_ABLATION_MODEL_PALETTE[model]
        for model in model_names
        if model in PUBLICATION_ABLATION_MODEL_PALETTE
    }


def _benchmark_palette(model_names: list[str]) -> dict[str, str | tuple[float, float, float, float]]:
    palette = dict(SCIB_BENCHMARK_COLORS)
    alias_model_names = [
        model_name for model_name in model_names if model_name in set(ANNOTATION_COLOR_ORDER)
    ]
    alias_palette = _annotation_alias_palette(alias_model_names)
    palette.update(alias_palette)
    return palette


def _annotation_alias_palette(
    model_names: list[str],
) -> dict[str, tuple[float, float, float, float]]:
    full_palette = annotation_ordered_blue_model_palette(ANNOTATION_COLOR_ORDER)
    fallback_palette = _ordered_palette(model_names)
    return {
        model_name: full_palette.get(model_name, fallback_palette.get(model_name, SCIB_BENCHMARK_COLORS["ML"]))
        for model_name in model_names
    }


def _ordered_model_labels_by_total(
    score_df: pd.DataFrame,
    *,
    model_labels: list[str],
) -> list[str]:
    return ordered_model_labels_by_metric(
        score_df,
        model_labels=model_labels,
        metric="Total",
        model_column="model_name",
        ascending=False,
    )


def _annotation_balanced_accuracy_order(model_labels: list[str]) -> list[str]:
    present_model_labels = [label for label in model_labels if label in set(ANNOTATION_COLOR_ORDER)]
    if not present_model_labels:
        return model_labels

    annotation_specs = [
        spec for spec in ANNOTATION_MODEL_SPECS if spec.label in set(present_model_labels)
    ]
    try:
        metrics = concat_available_annotation_frames(
            [load_annotation_metrics_summary(spec) for spec in annotation_specs],
            description="annotation metrics for scIB plot ordering",
        )
    except FileNotFoundError:
        return [label for label in ANNOTATION_COLOR_ORDER if label in set(model_labels)]

    ordered = ordered_model_labels_by_metric(
        metrics,
        model_labels=present_model_labels,
        metric="balanced_accuracy",
        model_column="plot_model",
        ascending=False,
    )
    ordered.extend(label for label in model_labels if label not in set(ordered))
    return ordered


def _set_plot_style() -> None:
    sns.set_theme(
        context="paper",
        style="white",
        rc={
            "axes.edgecolor": "0.2",
            "axes.linewidth": 0.7,
            "axes.grid": False,
            "font.size": 9,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
        },
    )


def _save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("pdf", "png", "svg"):
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(OUTPUT_DIR / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def _scib_table_display_label(model_name: str, *, publication_alias: bool = True) -> str:
    model_name = str(model_name)
    if publication_alias and model_name in set(ANNOTATION_COLOR_ORDER):
        return publication_model_label(model_name)
    return SCIB_TABLE_DISPLAY_LABELS.get(model_name, model_name)


def _aggregate_scores(scib_results: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in AGGREGATE_SCORE_COLUMNS if column not in scib_results.columns]
    if missing:
        raise KeyError(f"scIB results are missing aggregate score columns: {missing}")

    score_df = scib_results.drop(index="Metric Type", errors="ignore").copy()
    score_df = score_df[AGGREGATE_SCORE_COLUMNS].reset_index(names="obsm_key")
    score_df = score_df.merge(
        mapping,
        on="obsm_key",
        how="inner",
        validate="one_to_one",
    )
    return score_df


def _plot_model_score_lines(
    score_df: pd.DataFrame,
    *,
    model_order: list[str],
    palette: dict[str, str | tuple[float, float, float, float]],
    stem: str,
    title: str,
) -> None:
    plot_df = score_df[score_df["model_name"].isin(model_order)].copy()
    plot_df["model_name"] = pd.Categorical(
        plot_df["model_name"],
        categories=model_order,
        ordered=True,
    )
    plot_df = plot_df.sort_values("model_name", kind="mergesort")
    x_base = np.arange(len(model_order)) * SCIB_MODEL_X_SPACING

    fig, ax = plt.subplots(figsize=ANNOTATION_REFERENCE_FIGSIZE)
    for _, row in plot_df.iterrows():
        x_center = x_base[model_order.index(str(row["model_name"]))]
        xs = [x_center + SCIB_SCORE_OFFSETS[column] for column in AGGREGATE_SCORE_COLUMNS]
        ys = [float(row[column]) for column in AGGREGATE_SCORE_COLUMNS]
        ax.plot(xs, ys, color="0.35", linewidth=0.55, zorder=1)
        for column, x_value, y_value in zip(AGGREGATE_SCORE_COLUMNS, xs, ys):
            ax.scatter(
                x_value,
                y_value,
                s=22,
                color=SCORE_COLOR_MAP[column],
                edgecolor="white",
                linewidth=0.3,
                zorder=2,
                label=column,
            )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(),
        unique.keys(),
        frameon=False,
        loc="upper right",
        labelspacing=0.12,
        handlelength=1.0,
        handletextpad=0.35,
        borderaxespad=0.15,
        markerscale=0.85,
    )
    ax.set_xticks(x_base)
    ax.set_xticklabels(publication_model_labels(model_order), rotation=45, ha="right")
    ax.tick_params(axis="x", which="major", bottom=True, length=3, width=0.7, color="0.2")
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("scIB score")
    ax.set_xlabel("")
    ax.set_title(title, fontsize=9.5)
    if len(x_base):
        ax.set_xlim(min(x_base) - 0.24, max(x_base) + 0.24)
    sns.despine(ax=ax)
    _save_figure(fig, stem)


def _plot_score_axis_lines(
    score_df: pd.DataFrame,
    *,
    model_order: list[str],
    palette: dict[str, str | tuple[float, float, float, float]],
    stem: str,
) -> None:
    plot_df = score_df[score_df["model_name"].isin(model_order)].copy()
    x_positions = np.arange(len(AGGREGATE_SCORE_COLUMNS)) * 0.36

    fig, ax = plt.subplots(figsize=(2.25, 1.9))
    for model_name in model_order:
        row = plot_df[plot_df["model_name"] == model_name]
        if row.empty:
            continue
        ys = [float(row.iloc[0][column]) for column in AGGREGATE_SCORE_COLUMNS]
        ax.plot(
            x_positions,
            ys,
            marker="o",
            linewidth=1.0,
            markersize=3.5,
            color=palette.get(model_name, "0.5"),
            label=publication_model_label(model_name),
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(AGGREGATE_SCORE_COLUMNS, rotation=35, ha="right")
    ax.tick_params(axis="x", which="major", bottom=True, length=3, width=0.7, color="0.2")
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("scIB score")
    ax.set_xlabel("")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5))
    if len(x_positions):
        ax.set_xlim(min(x_positions) - 0.22, max(x_positions) + 0.22)
    sns.despine(ax=ax)
    _save_figure(fig, stem)


def _plot_total_barplot(
    score_df: pd.DataFrame,
    *,
    model_order: list[str],
    palette: dict[str, str | tuple[float, float, float, float]],
) -> None:
    plot_df = score_df[score_df["model_name"].isin(model_order)].copy()
    plot_df["model_name"] = pd.Categorical(
        plot_df["model_name"],
        categories=model_order,
        ordered=True,
    )
    plot_df = plot_df.sort_values("model_name", kind="mergesort")
    x_positions = np.arange(len(plot_df)) * BENCHMARK_BAR_X_SPACING

    fig, ax = plt.subplots(figsize=(max(2.45, len(plot_df) * 0.28 + 0.95), 1.75))
    ax.bar(
        x_positions,
        plot_df["Total"].astype(float),
        width=BENCHMARK_BAR_WIDTH,
        color=[palette[str(model)] for model in plot_df["model_name"]],
        edgecolor="0.15",
        linewidth=0.35,
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        publication_model_labels(plot_df["model_name"].astype(str).tolist()),
        rotation=45,
        ha="right",
    )
    ax.tick_params(axis="x", which="major", bottom=True, length=3, width=0.7, color="0.2")
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Total scIB score")
    ax.set_xlabel("")
    if len(x_positions):
        ax.set_xlim(min(x_positions) - 0.24, max(x_positions) + 0.24)
    sns.despine(ax=ax)
    _save_figure(fig, "benchmark_total_scib_score")


def _write_plots(score_df: pd.DataFrame) -> None:
    _set_plot_style()
    configured_ablation_order = [
        model
        for model in ABLATION_ORDER
        if model in set(score_df.loc[score_df["include_ablation"], "model_name"])
    ]
    configured_benchmark_order = [
        model
        for model in BENCHMARK_ORDER
        if model in set(score_df.loc[score_df["include_benchmark"], "model_name"])
    ]
    ablation_order = configured_ablation_order
    benchmark_order = _ordered_model_labels_by_total(
        score_df[score_df["include_benchmark"]],
        model_labels=configured_benchmark_order,
    )
    ablation_palette = _annotation_alias_palette(configured_ablation_order)
    benchmark_palette = _benchmark_palette(benchmark_order)

    ablation_df = score_df[score_df["include_ablation"]].copy()
    benchmark_df = score_df[score_df["include_benchmark"]].copy()

    if ablation_order:
        _plot_model_score_lines(
            ablation_df,
            model_order=ablation_order,
            palette=ablation_palette,
            stem="ablation_scib_scores_by_model",
            title="Ablation scIB aggregate scores",
        )
        _plot_score_axis_lines(
            ablation_df,
            model_order=ablation_order,
            palette=ablation_palette,
            stem="ablation_scib_scores_by_score",
        )
    if benchmark_order:
        _plot_total_barplot(
            benchmark_df,
            model_order=benchmark_order,
            palette=benchmark_palette,
        )
        _plot_model_score_lines(
            benchmark_df,
            model_order=benchmark_order,
            palette=benchmark_palette,
            stem="benchmark_scib_scores_by_model",
            title="Benchmark scIB aggregate scores",
        )


def _write_metadata(
    *,
    adata: ad.AnnData,
    mapping: pd.DataFrame,
    source_batch_key: str,
    skipped_models: list[str],
) -> None:
    metadata = {
        "dataset_id": DATASET_ID,
        "dataset_dir": str(DATASET_DIR),
        "adata_test_path": str(ADATA_TEST_PATH),
        "scrna_test_path": str(SCRNA_TEST_PATH),
        "scrna_hf_dataset": SCRNA_HF_DATASET,
        "annotation_column": ANNOTATION_COLUMN,
        "label_key": LABEL_KEY,
        "batch_key": BATCH_KEY,
        "source_batch_key": source_batch_key,
        "n_obs": int(adata.n_obs),
        "n_labels": int(adata.obs[LABEL_KEY].nunique()),
        "n_batches": int(adata.obs[BATCH_KEY].nunique()),
        "n_jobs": N_JOBS,
        "batch_size": BATCH_SIZE,
        "max_cells": MAX_CELLS,
        "pca_n_components": PCA_N_COMPONENTS,
        "force_all": FORCE_ALL,
        "force_scib_scores": FORCE_SCIB_SCORES,
        "force_embeddings": FORCE_EMBEDDINGS,
        "save_adata": SAVE_ADATA,
        "embedding_obsm_keys": mapping["obsm_key"].tolist(),
        "embedding_mapping_path": str(EMBEDDING_MAPPING_PATH),
        "selected_models": mapping["model_name"].tolist(),
        "skipped_models": skipped_models,
        "model_filter_env": MODEL_FILTER_ENV,
        "dependency_guidance": {
            "scib_metrics": 'uv pip install --python .venv_annotation/bin/python "scib-metrics>=0.5.0"',
            "pca_and_parquet": 'uv pip install --python .venv_annotation/bin/python "scikit-learn>=1.4" "pyarrow>=15"',
            "scvi_tools_optional": 'uv pip install --python .venv_annotation/bin/python "scvi-tools>=1.2"',
            "scgpt_optional": 'uv pip install --python .venv_annotation/bin/python "scgpt"',
        },
    }
    with (OUTPUT_DIR / "scib_run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _save_native_scib_results_table(
    bm,
    *,
    stem: str,
    model_to_obsm_key: dict[str, str] | None = None,
    model_order: list[str] | None = None,
    publication_labels: bool = False,
) -> None:
    """Save the native scib-metrics result table, optionally filtered to model groups."""
    original_results = bm._results
    original_embedding_obsm_keys = bm._embedding_obsm_keys
    try:
        if model_to_obsm_key is not None and model_order is not None:
            selected_models = [model for model in model_order if model in model_to_obsm_key]
            selected_obsm_keys = [model_to_obsm_key[model] for model in selected_models]
            selected_columns = [
                column for column in selected_obsm_keys if column in original_results.columns
            ]
            if not selected_columns:
                return
            selected_columns.append("Metric Type")
            subset = original_results.loc[:, selected_columns].copy()
            rename_map = {}
            for model in selected_models:
                obsm_key = model_to_obsm_key[model]
                if obsm_key not in subset.columns:
                    continue
                rename_map[obsm_key] = _scib_table_display_label(
                    model,
                    publication_alias=publication_labels or model in set(ANNOTATION_COLOR_ORDER),
                )
            subset = subset.rename(columns=rename_map)
            bm._results = subset
            bm._embedding_obsm_keys = selected_columns[:-1]
        else:
            rename_map = {
                column: _scib_table_display_label(column)
                for column in original_results.columns
                if column in SCIB_TABLE_DISPLAY_LABELS or column in set(ANNOTATION_COLOR_ORDER)
            }
            if rename_map:
                bm._results = original_results.rename(columns=rename_map)

        table = bm.plot_results_table(show=False, save_dir=None)
        figure = getattr(getattr(table, "ax", None), "figure", None)
        if figure is None:
            figure = plt.gcf()
        for suffix in ("pdf", "png", "svg"):
            kwargs = {"bbox_inches": "tight"}
            if suffix == "png":
                kwargs["dpi"] = 300
            figure.savefig(OUTPUT_DIR / f"{stem}.{suffix}", **kwargs)
        plt.close(figure)
    finally:
        bm._results = original_results
        bm._embedding_obsm_keys = original_embedding_obsm_keys


def _save_display_results_table(display_df: pd.DataFrame, stem: str) -> None:
    if display_df.empty or "Model" not in display_df.columns:
        return
    display_df = display_df.copy()
    display_df.to_csv(OUTPUT_DIR / f"{stem}.csv", index=False)

    value_columns = [column for column in display_df.columns if column != "Model"]
    for column in value_columns:
        display_df[column] = pd.to_numeric(display_df[column], errors="coerce").map(
            lambda value: f"{value:.3f}" if pd.notna(value) else ""
        )

    fig_width = max(3.6, 0.44 * len(display_df) + 2.1)
    fig_height = max(1.4, 0.28 * len(display_df) + 0.65)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    column_widths = [0.34, *([0.22] * len(value_columns))]
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=column_widths[: len(display_df.columns)],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.3)
    table.scale(1.0, 1.18)
    for (row, column), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor(SCIB_TABLE_HEADER_COLOR)
        if column == 0:
            cell.set_text_props(ha="left")
    for suffix in ("pdf", "png", "svg"):
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(OUTPUT_DIR / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def _refresh_cached_scib_results_tables() -> None:
    if not SCIB_AGGREGATE_SCORES_PATH.exists():
        return
    scores = pd.read_csv(SCIB_AGGREGATE_SCORES_PATH)
    required = {"model_name", "Batch correction", "Bio conservation", "Total"}
    if not required.issubset(scores.columns):
        return

    ablation_models = [
        model
        for model in ABLATION_ORDER
        if model in set(scores.loc[scores["include_ablation"], "model_name"])
    ]
    benchmark_models = _ordered_model_labels_by_total(
        scores[scores["include_benchmark"]],
        model_labels=[
            model
            for model in BENCHMARK_ORDER
            if model in set(scores.loc[scores["include_benchmark"], "model_name"])
        ],
    )
    for stem, model_order in {
        "ablation_scib_results_table": ablation_models,
        "benchmark_scib_results_table": benchmark_models,
    }.items():
        table_df = (
            scores[scores["model_name"].isin(model_order)]
            .set_index("model_name")
            .loc[model_order, ["Batch correction", "Bio conservation", "Total"]]
            .reset_index()
            .rename(columns={"model_name": "Model"})
        )
        table_df["Model"] = table_df["Model"].map(_scib_table_display_label)
        for column in ["Batch correction", "Bio conservation", "Total"]:
            table_df[column] = pd.to_numeric(table_df[column], errors="coerce").round(3)
        _save_display_results_table(table_df, stem)


def _load_scib_benchmark_classes():
    try:
        from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation
    except ImportError as error:
        raise ImportError(
            "scib_metrics is required for this script. Install it with:\n"
            'uv pip install --python .venv_annotation/bin/python "scib-metrics>=0.5.0"'
        ) from error
    return BatchCorrection, Benchmarker, BioConservation


def _run_scib(
    adata: ad.AnnData,
    mapping: pd.DataFrame,
    *,
    scib_classes,
) -> pd.DataFrame:
    BatchCorrection, Benchmarker, BioConservation = scib_classes
    bm = Benchmarker(
        adata,
        batch_key=BATCH_KEY,
        label_key=LABEL_KEY,
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
        embedding_obsm_keys=mapping["obsm_key"].tolist(),
        n_jobs=N_JOBS,
    )
    bm.benchmark()

    raw_results = bm.get_results(min_max_scale=False)
    scaled_results = bm.get_results(min_max_scale=True)
    raw_results.to_csv(SCIB_RAW_RESULTS_PATH)
    scaled_results.to_csv(SCIB_SCALED_RESULTS_PATH)
    aggregate_scores = _aggregate_scores(raw_results, mapping)
    model_to_obsm_key = (
        mapping.drop_duplicates(subset=["model_name"], keep="first")
        .set_index("model_name")["obsm_key"]
        .astype(str)
        .to_dict()
    )
    ablation_order = [
        model
        for model in ABLATION_ORDER
        if model in set(aggregate_scores.loc[aggregate_scores["include_ablation"], "model_name"])
    ]
    benchmark_order = _ordered_model_labels_by_total(
        aggregate_scores[aggregate_scores["include_benchmark"]],
        model_labels=[
            model
            for model in BENCHMARK_ORDER
            if model in set(aggregate_scores.loc[aggregate_scores["include_benchmark"], "model_name"])
        ],
    )

    _save_native_scib_results_table(bm, stem="scib_results_table")
    _save_native_scib_results_table(
        bm,
        stem="ablation_scib_results_table",
        model_to_obsm_key=model_to_obsm_key,
        model_order=ablation_order,
        publication_labels=True,
    )
    _save_native_scib_results_table(
        bm,
        stem="benchmark_scib_results_table",
        model_to_obsm_key=model_to_obsm_key,
        model_order=benchmark_order,
        publication_labels=False,
    )

    return aggregate_scores


def main() -> None:
    print_dependency_guidance()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_specs = _selected_embedding_specs()

    cached_scores = _load_cached_scib_scores_if_complete(selected_specs)
    if cached_scores is not None:
        print(f"Reusing cached complete scIB scores from {SCIB_AGGREGATE_SCORES_PATH}")
        _write_score_outputs(cached_scores, write_aggregate=False)
        _write_cached_run_metadata(specs=selected_specs, aggregate_scores=cached_scores)
        _write_plots(cached_scores)
        _refresh_cached_scib_results_tables()
        print(f"Saved scIB embedding comparison plots from cached scores to {OUTPUT_DIR}")
        return

    scib_classes = _load_scib_benchmark_classes()

    scrna_test = _load_scrna_test()
    scrna_rows = _select_scrna_rows(scrna_test)
    adata = _subset_adata_to_rows(_load_adata_test(), scrna_rows)
    adata, source_batch_key = _ensure_scib_obs_columns(adata)

    mapping_rows = []
    skipped_models = []
    for spec in selected_specs:
        adata, mapping_row = _append_embedding(adata, scrna_rows, spec)
        if mapping_row is None:
            skipped_models.append(spec.name)
            continue
        mapping_rows.append(mapping_row)

    if not mapping_rows:
        raise ValueError("No embeddings were prepared for scIB evaluation.")

    mapping = pd.DataFrame(mapping_rows)
    mapping.to_csv(EMBEDDING_MAPPING_PATH, index=False)
    _write_metadata(
        adata=adata,
        mapping=mapping,
        source_batch_key=source_batch_key,
        skipped_models=skipped_models,
    )

    if SAVE_ADATA:
        adata_path = OUTPUT_DIR / "adata_test_with_embedding_comparison.h5ad"
        adata.write_h5ad(adata_path, compression="gzip")
        print(f"Saved AnnData with embedding comparison obsm keys to {adata_path}")

    aggregate_scores = _run_scib(adata, mapping, scib_classes=scib_classes)
    _write_score_outputs(aggregate_scores)
    _write_plots(aggregate_scores)
    print(f"Saved scIB embedding comparison outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except ImportError as error:
        print(error)
        raise SystemExit(1) from error
