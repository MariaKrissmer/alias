from __future__ import annotations

import argparse
from dataclasses import dataclass
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

from alias.evaluation import (  # noqa: E402
    FunctionalityDownstreamConfig,
    run_functionality_downstream_assessment,
)


DATASET_ID = "S7_heldout_donor_semantic_cmv_200k"
REVISION_DATA_ROOT = Path(
    os.environ.get("ALIAS_REVISION_DATA_ROOT", PROJECT_ROOT / "out" / "data" / "revision1_v1")
)
DATASET_DIR = REVISION_DATA_ROOT / "HIAI_Tcells" / DATASET_ID
EMBEDDING_ROOT = (
    Path(os.environ["HIAI_TCELLS_CMV_EMBEDDING_ROOT"])
    if "HIAI_TCELLS_CMV_EMBEDDING_ROOT" in os.environ
    else None
)
OUTPUT_DIR = Path(
    os.environ.get(
        "HIAI_TCELLS_CMV_OUTPUT_DIR",
        DATASET_DIR / "analysis_runs" / "cmv_functionality",
    )
)
ADATA_PATH = DATASET_DIR / "adata_test.h5ad"
ANNOTATION_COLUMN = "AIFI_L2"
GROUND_TRUTH_COLUMN = "subject.cmv"
EVALUATION_ROOT = Path(
    os.environ.get("HIAI_TCELLS_CMV_EVALUATION_ROOT", OUTPUT_DIR / "evaluation_plots")
)
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
    model_id: str


MODEL_SPECS = {
    "MB": ModelSpec(
        label="MB",
        artifact_name="MB",
        model_id="mariakrissmer/ALIAS-MA",
    ),
    "MM": ModelSpec(
        label="MM",
        artifact_name="MM",
        model_id="mariakrissmer/ALIAS-ME",
    ),
}
DEFAULT_MODELS = ("MB", "MM")


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


def _rekey_embeddings_dict(embeddings_dict: dict, model_id: str, artifact_name: str) -> dict:
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
    if EMBEDDING_ROOT is None:
        raise RuntimeError("Set HIAI_TCELLS_CMV_EMBEDDING_ROOT to a fresh analysis run directory.")
    model_source = spec.model_id
    print(f"Generating S7 cell and disease-prompt embeddings for {model_label}: {model_source}")
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
    force_regenerate: bool = False,
) -> dict:
    selected = model_names or list(DEFAULT_MODELS)
    embeddings: dict = {}
    for label in selected:
        spec = MODEL_SPECS[label]
        generated = _generate_matching_disease_embeddings(
            label,
            disease_strings,
            force_regenerate=force_regenerate,
        )
        embeddings[label] = generated[spec.artifact_name]
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


def _model_name_map(model_names: list[str] | None = None) -> dict[str, str]:
    selected = model_names or list(DEFAULT_MODELS)
    return {label: MODEL_SPECS[label].model_id for label in selected}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run supervised CMV disease functionality assessment.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        help="Model labels to include. Defaults to: MB MM.",
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
        "--force-regenerate-disease-embeddings",
        action="store_true",
        help="Regenerate S7 embeddings even if the current analysis run already contains them.",
    )
    parser.add_argument(
        "--no-subject-level-deg",
        action="store_true",
        help="Skip subject-level pseudobulk DESeq2 for the supervised CMV comparison.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify S7 inputs and public model IDs without generating embeddings or results.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not SCRNA_TEST_PATH.is_dir():
        raise FileNotFoundError(f"Missing local S7 scrna_test dataset: {SCRNA_TEST_PATH}")
    if not ADATA_PATH.is_file():
        raise FileNotFoundError(f"Missing local S7 AnnData file: {ADATA_PATH}")
    if args.check:
        dataset = _load_scrna_test()
        missing_columns = {"index", "sentence1", ANNOTATION_COLUMN} - set(dataset.column_names)
        if missing_columns:
            raise ValueError(f"S7 scrna_test is missing columns: {sorted(missing_columns)}")
        print(f"S7 test dataset: {len(dataset)} cells at {SCRNA_TEST_PATH}")
        for label in args.models or DEFAULT_MODELS:
            print(f"{label}: {MODEL_SPECS[label].model_id}")
        print("Embeddings will be generated in a new analysis run.")
        return
    if EMBEDDING_ROOT is None or not (EMBEDDING_ROOT.parent / "run_manifest.tsv").is_file():
        raise RuntimeError(
            "Run scripts/revision1_v1/HIAI_Tcells/run_cmv_results.sh so embeddings "
            "are generated inside a new analysis_runs folder."
        )
    disease_strings = [str(value) for value in args.disease_strings]
    embeddings_dict = _load_embeddings(
        args.models,
        disease_strings=disease_strings,
        force_regenerate=args.force_regenerate_disease_embeddings,
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
