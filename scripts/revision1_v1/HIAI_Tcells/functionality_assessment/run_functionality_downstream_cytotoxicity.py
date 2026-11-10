from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
HIAI_TCELLS_SCRIPT_DIR = PROJECT_ROOT / "scripts" / "revision1_v1" / "HIAI_Tcells"
if str(HIAI_TCELLS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(HIAI_TCELLS_SCRIPT_DIR))

from alias.evaluation.functionality_downstream_assessment import (  # noqa: E402
    CELL_PROLIFERATION_GENE_SET,
    CYTOTOXIC_GENE_SET,
    FunctionalityDownstreamConfig,
    REGULATORY_GENE_SET,
    run_functionality_downstream_assessment,
)
from util.publication_plotting import PUBLICATION_ABLATION_MODEL_ORDER  # noqa: E402


DATASET_ID = "S2_heldout_donor_semantic_200k"
REVISION_DATA_ROOT = Path(
    os.environ.get("ALIAS_REVISION_DATA_ROOT", PROJECT_ROOT / "out" / "data" / "revision1_v1")
)
DATASET_DIR = REVISION_DATA_ROOT / "HIAI_Tcells" / DATASET_ID
EMBEDDING_ROOT = DATASET_DIR / "embeddings"
OUTPUT_DIR = Path(
    os.environ.get(
        "HIAI_TCELLS_FUNCTIONALITY_DOWNSTREAM_OUTPUT_DIR",
        DATASET_DIR / "functionality_assessment" / "downstream",
    )
)
ADATA_PATH = DATASET_DIR / "adata_test.h5ad"
ANNOTATION_COLUMN = "AIFI_L2"
EVALUATION_ROOT = Path(
    os.environ.get("HIAI_TCELLS_FUNCTIONALITY_EVALUATION_ROOT", DATASET_DIR / "evaluation_plots")
)
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
        model_id="mariakrissmer/ALIAS-MA",
    ),
    "Base": ModelSpec(
        label="Base",
        artifact_name="Base",
        model_id="neuml/pubmedbert-base-embeddings",
    ),
    "MG": ModelSpec(
        label="MG",
        artifact_name="MG_HIAI_Tcells_S2_200k_lr5e5_all",
        model_id="mariakrissmer/ALIAS-MB",
    ),
    "MI": ModelSpec(
        label="MI",
        artifact_name="MI_HIAI_Tcells_N1_200k_lr5e5_all",
        model_id="mariakrissmer/ALIAS-MD",
    ),
    "MJ": ModelSpec(
        label="MJ",
        artifact_name="MJ_HIAI_Tcells_S2_N3_200k_lr5e5",
        model_id="mariakrissmer/ALIAS-MA-star",
    ),
    "MF": ModelSpec(
        label="MF",
        artifact_name="MF_HIAI_Tcells_S3_200k_lr5e5_all",
        model_id="mariakrissmer/ALIAS-MC",
    ),
    "MH": ModelSpec(
        label="MH",
        artifact_name="MH_HIAI_Tcells_S5_200k_lr5e5",
        model_id="mariakrissmer/ALIAS-MC-star",
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


def _load_embeddings(
    model_names: list[str] | None = None,
    *,
    functionality_labels: list[str],
    generate_missing: bool = True,
    force_regenerate: bool = False,
    n_neighbors: int = 15,
    min_dist: float = 0.5,
) -> dict:
    from alias import evaluation as ev
    from alias.evaluation.functionality_benchmark import (
        attach_cached_functionality_umap,
        prepare_alias_functionality_embeddings,
    )

    selected = model_names or list(DEFAULT_ABLATION_MODELS)
    embeddings = {}
    for label in selected:
        spec = MODEL_SPECS[label]
        prepared, _, _ = prepare_alias_functionality_embeddings(
            dataset_id=DATASET_ID,
            model_label=label,
            model_id=str(spec.model_id),
            descriptions=functionality_labels,
            embedding_root=EMBEDDING_ROOT,
            annotation_column=ANNOTATION_COLUMN,
            max_cells=MAX_CELLS,
            batch_size=BATCH_SIZE,
            force_regenerate=force_regenerate,
            require_text_cache=not generate_missing,
        )
        if not attach_cached_functionality_umap(prepared, label):
            prepared = ev.umap_plots(
                embeddings_dict=prepared,
                annotation_column=ANNOTATION_COLUMN,
                output_dir=str(EVALUATION_ROOT),
                evaluation_config=ev.EvaluationConfig(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    random_state=21,
                    n_components=25,
                ),
            )
        embeddings[label] = prepared[label]
    return embeddings


def _model_name_map(model_names: list[str] | None = None) -> dict[str, str | Path]:
    selected = model_names or list(DEFAULT_ABLATION_MODELS)
    return {label: str(MODEL_SPECS[label].model_id) for label in selected}


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
            "Fail if no cached text embeddings match the exact Hugging Face model ID and "
            "functionality labels. Cell embeddings are always reused."
        ),
    )
    parser.add_argument(
        "--force-regenerate-functionality-embeddings",
        action="store_true",
        help="Regenerate functionality text embeddings; cached cell embeddings remain required.",
    )
    parser.add_argument(
        "--no-validate-functionality-embeddings",
        action="store_true",
        help=(
            "Compatibility option. Exact model ID and label metadata are always checked; "
            "cached text vectors are not re-encoded for validation."
        ),
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors used only when matching cached coordinates are absent.",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.5,
        help="UMAP min_dist used only when matching cached coordinates are absent.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.no_validate_functionality_embeddings:
        print("Cache reuse still requires matching Hugging Face model ID and exact functionality labels.")
    functionality_labels = FUNCTIONALITY_LABELS + SECOND_FUNCTIONALITY_LABELS
    gene_sets = {**GENE_SETS, **SECOND_GENE_SETS}
    embeddings_dict = _load_embeddings(
        args.models,
        functionality_labels=functionality_labels,
        generate_missing=not args.no_generate_missing_functionality_embeddings,
        force_regenerate=args.force_regenerate_functionality_embeddings,
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
