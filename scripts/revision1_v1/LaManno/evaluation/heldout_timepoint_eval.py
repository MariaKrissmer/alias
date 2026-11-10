from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from datasets import load_from_disk


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src" / "alias").exists():
            return parent
    raise RuntimeError("Could not resolve project root from this script location.")


PROJECT_ROOT = _find_project_root()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
LAMANNO_UTIL_PATH = PROJECT_ROOT / "scripts" / "revision1_v1" / "LaManno" / "util"
if str(LAMANNO_UTIL_PATH) not in sys.path:
    sys.path.insert(0, str(LAMANNO_UTIL_PATH))

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import scanpy as sc  # noqa: E402

from alias import evaluation as ev  # noqa: E402
from lineages import LAMANNO_LINEAGES  # noqa: E402


REVISION_DIR = Path(
    os.environ.get("ALIAS_REVISION_DATA_ROOT", PROJECT_ROOT / "out" / "data" / "revision1_v1")
) / "LaManno"
ANALYSIS_RUN_DIR = (
    Path(os.environ["LAMANNO_ANALYSIS_RUN_DIR"])
    if "LAMANNO_ANALYSIS_RUN_DIR" in os.environ
    else None
)
ANNOTATION_COLUMN = "celltype"
EVALUATION_DATASET_ID = "S9_heldout_sample_semantic"
HELDOUT_TIMEPOINT_DATASET_ID = "S10_heldout_timepoint_semantic"


@dataclass(frozen=True)
class LaMannoHeldoutTimepointSpec:
    model_label: str
    model_id: str


EVALUATION_SPECS: dict[str, LaMannoHeldoutTimepointSpec] = {
    "MP": LaMannoHeldoutTimepointSpec(
        model_label="MP",
        model_id="mariakrissmer/ALIAS-MG",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot S9 LaManno UMAPs with S10 held-out timepoints highlighted."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(EVALUATION_SPECS),
        choices=sorted(EVALUATION_SPECS),
    )
    parser.add_argument("--max-cells", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--force-regenerate", action="store_true")
    parser.add_argument("--n-neighbors", type=int, default=50)
    parser.add_argument("--min-dist", type=float, default=0.5)
    parser.add_argument("--n-components", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=73)
    parser.add_argument("--background-alpha", type=float, default=0.12)
    parser.add_argument("--highlighted-alpha", type=float, default=0.9)
    parser.add_argument("--quantification-k-neighbors", type=int, default=15)
    parser.add_argument("--only-neighborhood-assessment", action="store_true")
    return parser.parse_args()


def _heldout_timepoints_from_s10_metadata() -> list[str]:
    metadata_path = REVISION_DIR / HELDOUT_TIMEPOINT_DATASET_ID / "metadata" / "split_indices.json"
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    values = metadata.get("heldout_values", {}).get("timepoints")
    if not values:
        raise ValueError(f"Could not resolve held-out timepoints from {metadata_path}")
    return [str(float(value)) for value in values]


def _sample_overlap_audit(s9_adata, s10_adata) -> dict:
    s9_samples = set(s9_adata.obs["sample"].dropna().astype(str))
    s10_samples = set(s10_adata.obs["sample"].dropna().astype(str))
    return {
        "s9_test_dataset_id": EVALUATION_DATASET_ID,
        "s10_heldout_timepoint_dataset_id": HELDOUT_TIMEPOINT_DATASET_ID,
        "s9_test_samples": sorted(s9_samples),
        "s10_heldout_timepoint_test_samples": sorted(s10_samples),
        "shared_samples": sorted(s9_samples.intersection(s10_samples)),
        "s10_samples_not_in_s9_test": sorted(s10_samples.difference(s9_samples)),
        "s9_test_samples_not_in_s10": sorted(s9_samples.difference(s10_samples)),
        "same_sample_set": s9_samples == s10_samples,
    }


def _run_one_spec(
    spec: LaMannoHeldoutTimepointSpec,
    args: argparse.Namespace,
    heldout_timepoints: list[str],
) -> dict[str, str]:
    s9_dataset_dir = REVISION_DIR / EVALUATION_DATASET_ID
    s10_dataset_dir = REVISION_DIR / HELDOUT_TIMEPOINT_DATASET_ID
    output_dir = (
        ANALYSIS_RUN_DIR / HELDOUT_TIMEPOINT_DATASET_ID / spec.model_label / "heldout_timepoint_eval"
        if ANALYSIS_RUN_DIR is not None
        else s9_dataset_dir / "evaluation_plots" / "heldout_timepoint_eval"
    )
    embedding_output_dir = (
        ANALYSIS_RUN_DIR / "embeddings" / spec.model_label
        if ANALYSIS_RUN_DIR is not None
        else s9_dataset_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    model_source = spec.model_id
    model_source_metadata = {"source": "public_huggingface_model", "model_id": model_source}

    scrna_test = load_from_disk(str(s9_dataset_dir / "datasets" / "scrna_test"))
    s9_adata = sc.read_h5ad(s9_dataset_dir / "adata_test.h5ad")
    s10_adata = sc.read_h5ad(s10_dataset_dir / "adata_test.h5ad")

    embedding_config = ev.GenEmbeddingsConfig(
        annotation_column=ANNOTATION_COLUMN,
        embedding_models=[model_source],
        model_type="sentence_transformer",
        batch_size=args.batch_size,
        output_dir=str(embedding_output_dir),
        max_cells=args.max_cells,
        index=True,
        force_regenerate=args.force_regenerate,
    )
    evaluation_config = ev.EvaluationConfig(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.n_components,
        random_state=args.random_state,
    )
    embeddings_dict = ev.generate_embeddings(
        evaluation_dict={"scrna": {"test": scrna_test}},
        embedding_config=embedding_config,
    )
    if not args.only_neighborhood_assessment:
        print("Running held-out timepoint UMAP plots...", flush=True)
        ev.heldout_timepoint_umap_plots(
            embeddings_dict=embeddings_dict,
            adata=s9_adata,
            annotation_column=ANNOTATION_COLUMN,
            output_dir=output_dir,
            evaluation_config=evaluation_config,
            celltype_groups=LAMANNO_LINEAGES,
            heldout_timepoints=heldout_timepoints,
            background_alpha=args.background_alpha,
            highlighted_alpha=args.highlighted_alpha,
        )
    else:
        print("Skipping UMAP plots because --only-neighborhood-assessment was set.", flush=True)
    print("Running held-out timepoint neighborhood assessment...", flush=True)
    neighbor_metrics = ev.heldout_timepoint_neighbor_enrichment(
        embeddings_dict=embeddings_dict,
        eval_data={"adata": s9_adata},
        subfolder_fig_dir=output_dir,
        annotation_column=ANNOTATION_COLUMN,
        heldout_timepoints=heldout_timepoints,
        config=ev.HeldoutTimepointNeighborConfig(lineage="lineage_4", k_neighbors=args.quantification_k_neighbors),
    )
    neighbor_metrics_path = output_dir / f"{spec.model_label}_heldout_timepoint_neighbor_metrics.csv"
    neighbor_metrics.to_csv(neighbor_metrics_path, index=False)

    metadata = {
        "evaluation_name": "lamanno_heldout_timepoint_eval",
        "model": asdict(spec),
        "resolved_model_source": model_source,
        "model_source_metadata": model_source_metadata,
        "s9_dataset_dir": str(s9_dataset_dir),
        "s10_dataset_dir": str(s10_dataset_dir),
        "heldout_timepoints": heldout_timepoints,
        "sample_overlap_audit": _sample_overlap_audit(s9_adata, s10_adata),
        "embedding_config": asdict(embedding_config),
        "evaluation_config": asdict(evaluation_config),
        "neighbor_quantification": {
            "lineage": "lineage_4",
            "k_neighbors": args.quantification_k_neighbors,
            "metrics_path": str(neighbor_metrics_path),
        },
        "output_dir": str(output_dir),
    }
    metadata_path = output_dir / f"{spec.model_label}_heldout_timepoint_eval_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return {
        "heldout_timepoint_umap_dir": str(output_dir),
        "neighbor_metrics": str(neighbor_metrics_path),
        "metadata": str(metadata_path),
    }


def _selected_specs(model_labels: Iterable[str]) -> list[LaMannoHeldoutTimepointSpec]:
    return [EVALUATION_SPECS[model_label] for model_label in model_labels]


def main() -> None:
    args = _parse_args()
    heldout_timepoints = _heldout_timepoints_from_s10_metadata()
    outputs = {
        spec.model_label: _run_one_spec(spec, args, heldout_timepoints)
        for spec in _selected_specs(args.models)
    }
    print("Saved held-out timepoint UMAP outputs:")
    for model_label, model_outputs in outputs.items():
        print(f"- {model_label}")
        for output_name, output_path in model_outputs.items():
            print(f"  {output_name}: {output_path}")


if __name__ == "__main__":
    main()
