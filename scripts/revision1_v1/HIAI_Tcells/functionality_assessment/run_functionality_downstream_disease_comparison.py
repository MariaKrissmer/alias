from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alias.evaluation import (  # noqa: E402
    FunctionalityDownstreamConfig,
    run_functionality_downstream_assessment,
)


DATASET_ID = "S7_heldout_donor_semantic_cmv_200k"
DATASET_DIR = PROJECT_ROOT / "out" / "data" / "revision1_v1" / "HIAI_Tcells" / DATASET_ID
EMBEDDING_ROOT = DATASET_DIR / "embeddings"
OUTPUT_DIR = DATASET_DIR / "functionality_assessment" / "disease_comparison"
ADATA_PATH = DATASET_DIR / "adata_test.h5ad"
ANNOTATION_COLUMN = "AIFI_L2"
GROUND_TRUTH_COLUMN = "subject.cmv"
EVALUATION_ROOT = DATASET_DIR / "evaluation_plots"
SCRNA_TEST_PATH = DATASET_DIR / "datasets" / "scrna_test"
SCRNA_HF_DATASET = f"mariakrissmer/scrna_HIAI_Tcells_{DATASET_ID}"
MAX_CELLS = int(os.environ.get("FUNCTIONALITY_DISEASE_MAX_CELLS", "20000"))
BATCH_SIZE = int(os.environ.get("FUNCTIONALITY_DISEASE_BATCH_SIZE", "256"))

DISEASE_STRINGS = ["increased cytotoxic activity in cytomegalovirus positive patients"]
POSITIVE_VALUES = ["Positive", "positive", "CMV+", "cmv+", True, 1]


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
    "MK": ModelSpec(
        label="MK",
        artifact_name="MK_HIAI_Tcells_S7_N4_200k_lr5e5",
        model_id="mariakrissmer/MK_HIAI_Tcells_S7_N4_200k_lr5e5",
    ),
    "ML": ModelSpec(
        label="ML",
        artifact_name="ML_HIAI_Tcells_S7_N1_200k_lr5e5",
        model_id="mariakrissmer/ML_HIAI_Tcells_S7_N1_200k_lr5e5",
    ),
    "MM": ModelSpec(
        label="MM",
        artifact_name="MM_HIAI_Tcells_S2_N4_200k_lr5e5_epoch_9_ncbi",
        model_id=PROJECT_ROOT
        / "models"
        / "MM_HIAI_Tcells_S2_N4_200k_lr5e5_all"
        / "epoch_9_ncbi",
    ),
}
DEFAULT_MODELS = ("MB", "MK", "ML", "MM")


def _metadata_functionality_frame(metadata_path: Path):
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
    return set(labels).issubset(set(df_additional[label_column].astype(str)))


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
    if not set(labels).issubset(set(available.index)):
        return False

    numeric_columns = available.select_dtypes(include=[np.number]).columns
    saved = available.loc[labels, numeric_columns].to_numpy(dtype=float)
    model = load_embedding_model(str(_model_source(model_id)))
    encoded = np.asarray(
        model.encode(labels, batch_size=BATCH_SIZE, show_progress_bar=False),
        dtype=float,
    )
    if saved.shape != encoded.shape:
        return False
    close = np.allclose(saved, encoded, rtol=1e-4, atol=1e-5)
    if not close:
        diff = np.abs(saved - encoded)
        print(
            f"Skipping {metadata_path}: disease prompt embeddings do not match "
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
    candidates: list[Path] = []
    for root in [EMBEDDING_ROOT / "scrna" / artifact_name, EMBEDDING_ROOT / artifact_name]:
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
        return candidate
    return None


def _load_scrna_test():
    from datasets import load_from_disk
    from alias.util.load_hf_model import load_hf_dataset

    if SCRNA_TEST_PATH.exists():
        return load_from_disk(str(SCRNA_TEST_PATH))

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


def _model_source(model_id: str | Path) -> str | Path:
    model_path = Path(model_id)
    if model_path.exists():
        return str(model_path)
    local_path = PROJECT_ROOT / "models" / str(model_id)
    if local_path.exists():
        return str(local_path)
    return str(model_id)


def _rekey_embeddings_dict(embeddings_dict: dict, model_id: str | Path, artifact_name: str) -> dict:
    from alias.evaluation.embedding import clean_model_name

    source_key = clean_model_name(str(model_id))
    if source_key not in embeddings_dict:
        return embeddings_dict
    return {artifact_name: embeddings_dict[source_key]}


def _generate_matching_disease_embeddings(
    model_label: str,
    labels: list[str],
    *,
    force_regenerate: bool = False,
) -> dict:
    from alias import evaluation as ev

    spec = MODEL_SPECS[model_label]
    model_source = _model_source(spec.model_id)
    print(f"Generating matching disease-prompt embeddings for {model_label}: {model_source}")
    embeddings_dict = ev.generate_embeddings(
        evaluation_dict={"scrna": {"test": _load_scrna_test()}},
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
    return _rekey_embeddings_dict(
        embeddings_dict,
        model_id=model_source,
        artifact_name=spec.artifact_name,
    )


def _load_embeddings(
    model_names: list[str] | None = None,
    *,
    disease_strings: list[str],
    generate_missing: bool = True,
    force_regenerate: bool = False,
    validate_functionality_embeddings: bool = True,
) -> dict:
    selected = model_names or list(DEFAULT_MODELS)
    embeddings: dict = {}
    for label in selected:
        spec = MODEL_SPECS[label]
        metadata_path = None if force_regenerate else _matching_embedding_metadata(
            spec.artifact_name,
            disease_strings,
            model_id=spec.model_id,
            validate_functionality_embeddings=validate_functionality_embeddings,
        )
        if metadata_path is None:
            if not generate_missing:
                raise FileNotFoundError(
                    f"No embedding metadata for {label} contains exact disease strings: "
                    f"{disease_strings}"
                )
            generated = _generate_matching_disease_embeddings(
                label,
                disease_strings,
                force_regenerate=force_regenerate,
            )
            embeddings[label] = generated.get(spec.artifact_name, next(iter(generated.values())))
            continue
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        embeddings[label] = {metadata.get("dataset_name", "scrna"): metadata["artifacts"]}
    return embeddings


def _extra_plot_name(column: str) -> str:
    return f"cells_colored_by_{column.replace('.', '_')}.svg"


def _has_umap_paths(embeddings_dict: dict, *, extra_cell_annotation_columns: list[str] | None = None) -> bool:
    extra_cell_annotation_columns = extra_cell_annotation_columns or []
    for model_name, model_data in embeddings_dict.items():
        for dataset_name, dataset_meta in model_data.items():
            cell_meta = dataset_meta.get("df_cells", {})
            cell_umap = cell_meta.get("umap", {})
            cell_path = Path(cell_meta["path"]) if cell_meta.get("path") else None
            cell_umap_path = Path(cell_umap["path"]) if cell_umap.get("path") else None
            if cell_path is None or cell_umap_path is None or not cell_umap_path.exists():
                return False
            if cell_path.exists():
                import pandas as pd

                cell_index = pd.read_parquet(cell_path, columns=[]).index.astype(str)
                umap_index = pd.read_parquet(cell_umap_path, columns=[]).index.astype(str)
                if cell_index.intersection(umap_index).empty:
                    print(
                        f"Recomputing model-space UMAPs: {cell_umap_path} does not "
                        f"share cell IDs with {cell_path}."
                    )
                    return False
            figure_dir = (
                EVALUATION_ROOT
                / model_name
                / dataset_name
                / "celltype_label_plots"
                / cell_umap_path.parent.name
            )
            for column in extra_cell_annotation_columns:
                if not (figure_dir / _extra_plot_name(column)).exists():
                    print(f"Recomputing model-space UMAPs: missing {figure_dir / _extra_plot_name(column)}.")
                    return False
            if "df_celltypes" in dataset_meta:
                celltype_umap = dataset_meta.get("df_celltypes", {}).get("umap", {})
                if not celltype_umap.get("path") or not Path(celltype_umap["path"]).exists():
                    return False
    return True


def _extra_cell_annotations_from_adata(adata) -> dict[str, dict[str, str]]:
    if adata is None or GROUND_TRUTH_COLUMN not in adata.obs:
        return {}
    return {
        GROUND_TRUTH_COLUMN: adata.obs[GROUND_TRUTH_COLUMN].astype(str).to_dict(),
    }


def _ensure_model_umap_coordinates(embeddings_dict: dict, *, adata=None) -> dict:
    extra_cell_annotations = _extra_cell_annotations_from_adata(adata)
    if _has_umap_paths(
        embeddings_dict,
        extra_cell_annotation_columns=list(extra_cell_annotations),
    ):
        return embeddings_dict

    from alias import evaluation as ev

    print("Computing model-space UMAP coordinates for disease comparison embeddings.")
    return ev.umap_plots(
        embeddings_dict=embeddings_dict,
        annotation_column=ANNOTATION_COLUMN,
        output_dir=str(EVALUATION_ROOT),
        evaluation_config=ev.EvaluationConfig(
            n_neighbors=50,
            min_dist=0.25,
            random_state=21,
            n_components=25,
        ),
        extra_cell_annotations=extra_cell_annotations,
    )


def _model_name_map(model_names: list[str] | None = None) -> dict[str, str | Path]:
    selected = model_names or list(DEFAULT_MODELS)
    return {label: _model_source(MODEL_SPECS[label].model_id) for label in selected}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run supervised CMV disease functionality assessment.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        help="Model labels to include. Defaults to: MB MK ML MM.",
    )
    parser.add_argument(
        "--disease-strings",
        nargs="+",
        default=DISEASE_STRINGS,
        help="Disease/functionality prompt strings. Must exactly match saved df_additional labels when reuse is required.",
    )
    parser.add_argument(
        "--assessment-scope",
        choices=["per_cell_type", "all_cells"],
        default="per_cell_type",
        help="Run disease comparison per cell type or across all cells.",
    )
    parser.add_argument(
        "--no-generate-missing-disease-embeddings",
        action="store_true",
        help="Fail if matching disease-prompt embeddings are missing instead of generating them.",
    )
    parser.add_argument(
        "--force-regenerate-disease-embeddings",
        action="store_true",
        help="Regenerate disease-prompt embeddings even if matching metadata exists.",
    )
    parser.add_argument(
        "--no-validate-disease-embeddings",
        action="store_true",
        help="Reuse saved disease-prompt embeddings by label only, without comparing current model encodings.",
    )
    parser.add_argument(
        "--no-subject-level-deg",
        action="store_true",
        help="Skip subject-level pseudobulk DESeq2 for the supervised CMV comparison.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    disease_strings = [str(value) for value in args.disease_strings]
    embeddings_dict = _load_embeddings(
        args.models,
        disease_strings=disease_strings,
        generate_missing=not args.no_generate_missing_disease_embeddings,
        force_regenerate=args.force_regenerate_disease_embeddings,
        validate_functionality_embeddings=not args.no_validate_disease_embeddings,
    )
    adata = None
    if ADATA_PATH.exists():
        import scanpy as sc

        adata = sc.read_h5ad(ADATA_PATH)
    if not args.no_plots:
        embeddings_dict = _ensure_model_umap_coordinates(embeddings_dict, adata=adata)
    outputs = run_functionality_downstream_assessment(
        embeddings_dict=embeddings_dict,
        eval_data={"adata": adata} if adata is not None else None,
        config=FunctionalityDownstreamConfig(
            functionality_labels=disease_strings,
            output_dir=OUTPUT_DIR,
            adata_path=ADATA_PATH if adata is None and ADATA_PATH.exists() else None,
            annotation_column=ANNOTATION_COLUMN,
            ground_truth_column=GROUND_TRUTH_COLUMN,
            positive_values=POSITIVE_VALUES,
            assessment_scope=args.assessment_scope,
            cutoff_method="youden_j",
            gene_sets={},
            model_name_map=_model_name_map(args.models),
            supervised_downstream_plots=True,
            subject_level_deg=not args.no_subject_level_deg,
            subject_column="subject.subjectGuid",
            pseudobulk_layer="counts",
            min_subjects_per_group=2,
            plot=not args.no_plots,
        ),
    )
    print(f"Saved disease functionality assessment to {outputs['run_dir']}")


if __name__ == "__main__":
    main()
