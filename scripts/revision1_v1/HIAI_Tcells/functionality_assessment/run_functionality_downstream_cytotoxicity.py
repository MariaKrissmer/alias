from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
HIAI_TCELLS_SCRIPT_DIR = PROJECT_ROOT / "scripts" / "revision1_v1" / "HIAI_Tcells"
if str(HIAI_TCELLS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(HIAI_TCELLS_SCRIPT_DIR))

from alias.evaluation import (  # noqa: E402
    CELL_PROLIFERATION_GENE_SET,
    CYTOTOXIC_GENE_SET,
    FunctionalityDownstreamConfig,
    REGULATORY_GENE_SET,
    run_functionality_downstream_assessment,
)
from util.publication_plotting import PUBLICATION_ABLATION_MODEL_ORDER  # noqa: E402


DATASET_ID = "S2_heldout_donor_semantic_200k"
DATASET_DIR = PROJECT_ROOT / "out" / "data" / "revision1_v1" / "HIAI_Tcells" / DATASET_ID
EMBEDDING_ROOT = DATASET_DIR / "embeddings"
OUTPUT_DIR = DATASET_DIR / "functionality_assessment" / "downstream"
ADATA_PATH = DATASET_DIR / "adata_test.h5ad"
ANNOTATION_COLUMN = "AIFI_L2"
EVALUATION_ROOT = DATASET_DIR / "evaluation_plots"
SCRNA_TEST_PATH = DATASET_DIR / "datasets" / "scrna_test"
SCRNA_HF_DATASET = f"mariakrissmer/scrna_HIAI_Tcells_{DATASET_ID}"
MAX_CELLS = int(os.environ.get("FUNCTIONALITY_DOWNSTREAM_MAX_CELLS", "20000"))
BATCH_SIZE = int(os.environ.get("FUNCTIONALITY_DOWNSTREAM_BATCH_SIZE", "256"))


@dataclass(frozen=True)
class ModelSpec:
    label: str
    artifact_name: str
    model_id: str | Path


MODEL_SPECS = {
    "MB": ModelSpec(
        label="MB",
        artifact_name="MB_HIAI_Tcells_S2_N1_200k_lr5e5_e15_epoch_15_ncbi",
        model_id=PROJECT_ROOT
        / "models"
        / "MB_HIAI_Tcells_S2_N1_200k_lr5e5_e15_all"
        / "epoch_15_ncbi",
    ),
    "Base": ModelSpec(
        label="Base",
        artifact_name="Base",
        model_id="neuml/pubmedbert-base-embeddings",
    ),
    "MG": ModelSpec(
        label="MG",
        artifact_name="MG_HIAI_Tcells_S2_200k_lr5e5_all",
        model_id=PROJECT_ROOT / "models" / "MG_HIAI_Tcells_S2_200k_lr5e5_all",
    ),
    "MI": ModelSpec(
        label="MI",
        artifact_name="MI_HIAI_Tcells_N1_200k_lr5e5_all",
        model_id=PROJECT_ROOT / "models" / "MI_HIAI_Tcells_N1_200k_lr5e5_all",
    ),
    "MJ": ModelSpec(
        label="MJ",
        artifact_name="MJ_HIAI_Tcells_S2_N3_200k_lr5e5",
        model_id="mariakrissmer/MJ_HIAI_Tcells_S2_N3_200k_lr5e5",
    ),
    "MF": ModelSpec(
        label="MF",
        artifact_name="MF_HIAI_Tcells_S3_200k_lr5e5_all",
        model_id=PROJECT_ROOT / "models" / "MF_HIAI_Tcells_S3_200k_lr5e5_all",
    ),
    "MH": ModelSpec(
        label="MH",
        artifact_name="MH_HIAI_Tcells_S5_200k_lr5e5",
        model_id="mariakrissmer/MH_HIAI_Tcells_S5_200k_lr5e5",
    ),
}
DEFAULT_ABLATION_MODELS = tuple(PUBLICATION_ABLATION_MODEL_ORDER)

FUNCTIONALITY_LABELS = [
    "regulatory",
    "cytotoxic",
    "proliferating",
]
GENE_SETS = {
    "regulatory": REGULATORY_GENE_SET,
    "cytotoxic": CYTOTOXIC_GENE_SET,
    "proliferating": CELL_PROLIFERATION_GENE_SET,
}
SECOND_FUNCTIONALITY_LABELS: list[str] = []
SECOND_GENE_SETS: dict[str, list[str]] = {}


def _latest_embedding_metadata(artifact_name: str) -> Path:
    search_roots = [
        EMBEDDING_ROOT / "scrna" / artifact_name,
        EMBEDDING_ROOT / artifact_name,
    ]
    candidates: list[Path] = []
    for root in search_roots:
        if root.exists():
            candidates.extend(root.glob("**/metadata.json"))
    candidates = sorted(candidates, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No embedding metadata found for {artifact_name} under {EMBEDDING_ROOT}.")
    return candidates[0]


def _metadata_functionality_frame(metadata_path: Path):
    import json
    from alias.evaluation.embedding import load_dataset_embedding_artifacts

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    artifacts = metadata.get("artifacts", {})
    if "df_additional" not in artifacts:
        return None
    loaded = load_dataset_embedding_artifacts(artifacts, annotation_column=ANNOTATION_COLUMN)["artifacts"]
    additional = loaded.get("df_additional")
    if additional is None:
        return None
    return additional["dataframe"]


def _metadata_has_functionality_labels(metadata_path: Path, labels: list[str]) -> bool:
    df_additional = _metadata_functionality_frame(metadata_path)
    if df_additional is None:
        return False
    label_column = "data" if "data" in df_additional.columns else df_additional.columns[0]
    available = set(df_additional[label_column].astype(str))
    return set(labels).issubset(available)


def _metadata_has_current_functionality_embeddings(
    metadata_path: Path,
    labels: list[str],
    *,
    model_id: str | Path,
) -> bool:
    from alias.evaluation.embedding import load_embedding_model

    df_additional = _metadata_functionality_frame(metadata_path)
    if df_additional is None:
        return False
    label_column = "data" if "data" in df_additional.columns else df_additional.columns[0]
    available = df_additional.set_index(label_column)
    missing = [label for label in labels if label not in available.index]
    if missing:
        return False

    numeric_columns = available.select_dtypes(include=[np.number]).columns
    saved = available.loc[labels, numeric_columns].to_numpy(dtype=float)
    model = load_embedding_model(str(_model_source(model_id)))
    encoded = np.asarray(
        model.encode(labels, batch_size=BATCH_SIZE, show_progress_bar=False),
        dtype=float,
    )
    if saved.shape != encoded.shape:
        print(
            f"Skipping {metadata_path}: saved functionality embedding shape "
            f"{saved.shape} does not match current encoding shape {encoded.shape}."
        )
        return False
    close = np.allclose(saved, encoded, rtol=1e-4, atol=1e-5)
    if not close:
        diff = np.abs(saved - encoded)
        print(
            f"Skipping {metadata_path}: functionality embeddings do not match "
            f"current model encoding (max_abs_diff={diff.max():.4g}, "
            f"mean_abs_diff={diff.mean():.4g})."
        )
    return bool(close)


def _matching_embedding_metadata(
    artifact_name: str,
    labels: list[str],
    *,
    model_id: str | Path,
    validate_functionality_embeddings: bool = True,
) -> Path | None:
    search_roots = [
        EMBEDDING_ROOT / "scrna" / artifact_name,
        EMBEDDING_ROOT / artifact_name,
    ]
    candidates: list[Path] = []
    for root in search_roots:
        if root.exists():
            candidates.extend(root.glob("**/metadata.json"))
    for candidate in sorted(candidates, reverse=True):
        if not _metadata_has_functionality_labels(candidate, labels):
            continue
        if validate_functionality_embeddings and not _metadata_has_current_functionality_embeddings(
            candidate,
            labels,
            model_id=model_id,
        ):
            continue
        if _metadata_has_functionality_labels(candidate, labels):
            return candidate
    return None


def _load_scrna_test():
    from datasets import load_from_disk
    from alias.util.load_hf_model import load_hf_dataset

    if SCRNA_TEST_PATH.exists():
        try:
            return load_from_disk(str(SCRNA_TEST_PATH))
        except Exception as error:
            print(
                f"Could not load local scrna_test from {SCRNA_TEST_PATH}: "
                f"{type(error).__name__}: {error}"
            )
            print(f"Falling back to Hugging Face dataset: {SCRNA_HF_DATASET}")

    dataset = load_hf_dataset(SCRNA_HF_DATASET)
    if isinstance(dataset, dict):
        for split_name in ("test", "scrna_test"):
            if split_name in dataset:
                return dataset[split_name]
        raise KeyError(
            f"Could not find a test split in {SCRNA_HF_DATASET}. "
            f"Available splits: {sorted(dataset.keys())}"
        )
    return dataset


def _rekey_embeddings_dict(embeddings_dict: dict, model_id: str | Path, artifact_name: str) -> dict:
    from alias.evaluation.embedding import clean_model_name

    source_key = clean_model_name(str(model_id))
    if source_key not in embeddings_dict:
        return embeddings_dict
    return {artifact_name: embeddings_dict[source_key]}


def _generate_matching_functionality_embeddings(
    model_label: str,
    labels: list[str],
    *,
    force_regenerate: bool = False,
    n_neighbors: int = 15,
    min_dist: float = 0.5,
) -> dict:
    from alias import evaluation as ev

    spec = MODEL_SPECS[model_label]
    model_source = _model_source(spec.model_id)
    print(f"Generating matching functionality embeddings for {model_label}: {model_source}")
    scrna_test = _load_scrna_test()
    embeddings_dict = ev.generate_embeddings(
        evaluation_dict={"scrna": {"test": scrna_test}},
        embedding_config=ev.GenEmbeddingsConfig(
            embedding_models=[str(model_source)],
            model_type="sentence_transformer",
            batch_size=BATCH_SIZE,
            output_dir=str(EMBEDDING_ROOT / spec.artifact_name),
            max_cells=MAX_CELLS,
            annotation_column=ANNOTATION_COLUMN,
            additional_data=labels,
            force_regenerate=force_regenerate,
        ),
    )
    embeddings_dict = _rekey_embeddings_dict(
        embeddings_dict,
        model_id=model_source,
        artifact_name=spec.artifact_name,
    )
    embeddings_dict = ev.umap_plots(
        embeddings_dict=embeddings_dict,
        annotation_column=ANNOTATION_COLUMN,
        output_dir=str(EVALUATION_ROOT),
        evaluation_config=ev.EvaluationConfig(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=21,
            n_components=25,
        ),
    )
    ev.functionality_similarity(
        embeddings_dict=embeddings_dict,
        annotation_column=ANNOTATION_COLUMN,
        config=ev.FunctionalitySimilarityConfig(
            similarity_metric="cosine",
            bins=60,
            output_dir=EVALUATION_ROOT,
            plot=True,
        ),
    )
    return embeddings_dict


def _load_embeddings(
    model_names: list[str] | None = None,
    *,
    functionality_labels: list[str],
    generate_missing: bool = True,
    force_regenerate: bool = False,
    validate_functionality_embeddings: bool = True,
    n_neighbors: int = 15,
    min_dist: float = 0.5,
) -> dict:
    import json

    selected = model_names or list(DEFAULT_ABLATION_MODELS)
    embeddings: dict = {}
    for label in selected:
        artifact_name = MODEL_SPECS[label].artifact_name
        metadata_path = None if force_regenerate else _matching_embedding_metadata(
            artifact_name,
            functionality_labels,
            model_id=MODEL_SPECS[label].model_id,
            validate_functionality_embeddings=validate_functionality_embeddings,
        )
        if metadata_path is None:
            if not generate_missing:
                raise FileNotFoundError(
                    f"No embedding metadata for {label} contains exact functionality labels: "
                    f"{functionality_labels}"
                )
            generated = _generate_matching_functionality_embeddings(
                label,
                functionality_labels,
                force_regenerate=force_regenerate,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
            )
            embeddings[label] = generated.get(artifact_name, next(iter(generated.values())))
            continue
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        embeddings[label] = {metadata.get("dataset_name", "scrna"): metadata["artifacts"]}
    return embeddings


def _model_source(model_id: str | Path) -> str | Path:
    model_path = Path(model_id)
    if model_path.exists():
        return str(model_path)

    local_path = PROJECT_ROOT / "models" / str(model_id)
    if local_path.exists():
        return str(local_path)

    return str(model_id)


def _model_name_map(model_names: list[str] | None = None) -> dict[str, str | Path]:
    selected = model_names or list(DEFAULT_ABLATION_MODELS)
    return {label: _model_source(MODEL_SPECS[label].model_id) for label in selected}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run downstream cytotoxicity functionality assessment.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        help="Model labels to include. Defaults to all ablation models: MB MJ MG MF MH MI Base.",
    )
    parser.add_argument(
        "--cutoff-method",
        choices=["youden_j", "manual", "otsu", "gmm"],
        default="otsu",
        help="Cutoff method for assigning functionality-associated cells.",
    )
    parser.add_argument(
        "--assessment-scope",
        choices=["all_cells", "per_cell_type"],
        default="all_cells",
        help="Run one global assessment across all cells or one assessment per cell type.",
    )
    parser.add_argument(
        "--no-generate-missing-functionality-embeddings",
        action="store_true",
        help=(
            "Fail if no saved embedding metadata contains the exact requested functionality labels. "
            "By default, matching embeddings are generated first with functionality_similarity()."
        ),
    )
    parser.add_argument(
        "--force-regenerate-functionality-embeddings",
        action="store_true",
        help="Regenerate functionality-label embeddings before downstream assessment even if matching metadata exists.",
    )
    parser.add_argument(
        "--no-validate-functionality-embeddings",
        action="store_true",
        help=(
            "Reuse saved functionality embeddings based on metadata labels only. "
            "By default, saved df_additional vectors must match current model encodings."
        ),
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors used when missing functionality embeddings are generated.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.5,
        help="UMAP min_dist used when missing functionality embeddings are generated.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    functionality_labels = FUNCTIONALITY_LABELS + SECOND_FUNCTIONALITY_LABELS
    gene_sets = {**GENE_SETS, **SECOND_GENE_SETS}
    embeddings_dict = _load_embeddings(
        args.models,
        functionality_labels=functionality_labels,
        generate_missing=not args.no_generate_missing_functionality_embeddings,
        force_regenerate=args.force_regenerate_functionality_embeddings,
        validate_functionality_embeddings=not args.no_validate_functionality_embeddings,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
    )

    adata = None
    if ADATA_PATH.exists():
        import scanpy as sc

        adata = sc.read_h5ad(ADATA_PATH)
    outputs = run_functionality_downstream_assessment(
        embeddings_dict=embeddings_dict,
        eval_data={"adata": adata} if adata is not None else None,
        config=FunctionalityDownstreamConfig(
            functionality_labels=functionality_labels,
            output_dir=OUTPUT_DIR,
            adata_path=ADATA_PATH if adata is None and ADATA_PATH.exists() else None,
            annotation_column=ANNOTATION_COLUMN,
            assessment_scope=args.assessment_scope,
            cutoff_method=args.cutoff_method,
            gene_sets=gene_sets,
            model_name_map=_model_name_map(args.models),
            cutoff_sensitivity_threshold_max=0.6,
            plot=not args.no_plots,
        ),
    )
    print(f"Saved downstream functionality assessment to {outputs['run_dir']}")


if __name__ == "__main__":
    main()
