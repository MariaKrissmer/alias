from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
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
from lineages import (  # noqa: E402
    LAMANNO_GROUPS_L,
    LAMANNO_GROUPS_M_SPLIT,
    LAMANNO_LINEAGE_CELLTYPE_ORDER,
    LAMANNO_LINEAGES,
)


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


@dataclass(frozen=True)
class LaMannoEvaluationSpec:
    model_label: str
    dataset_id: str
    model_id: str


EVALUATION_SPECS: dict[str, LaMannoEvaluationSpec] = {
    "MO": LaMannoEvaluationSpec(
        model_label="MO",
        dataset_id=EVALUATION_DATASET_ID,
        model_id="mariakrissmer/ALIAS-MF",
    ),
    "MP": LaMannoEvaluationSpec(
        model_label="MP",
        dataset_id=EVALUATION_DATASET_ID,
        model_id="mariakrissmer/ALIAS-MG",
    ),
}
DEFAULT_MODEL_LABELS = ("MO", "MP")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the LaManno embedding evaluation wrapper."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODEL_LABELS),
        choices=sorted(EVALUATION_SPECS),
        help="LaManno model labels to evaluate.",
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        default=50000,
        help="Maximum cells passed to GenEmbeddingsConfig. Use 0 to disable subsampling.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Regenerate embeddings instead of reusing a matching cached embedding run.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--only-timepoint-similarity",
        action="store_true",
        help="Run only the LaManno lineage-wise timepoint similarity evaluation after loading/generating embeddings.",
    )
    mode_group.add_argument(
        "--only-umap",
        action="store_true",
        help="Run only grouped LaManno UMAP plots after loading/generating embeddings.",
    )
    parser.add_argument(
        "--lineage",
        default="lineage_4",
        help="Lineage passed to PseudotimeConfig.",
    )
    parser.add_argument(
        "--cell-origin",
        default="Forebrain",
        help="Root cell type passed to PseudotimeConfig.",
    )
    parser.add_argument("--n-neighbors", type=int, default=50)
    parser.add_argument("--min-dist", type=float, default=0.5)
    parser.add_argument("--n-components", type=int, default=50)
    parser.add_argument("--random-state", type=int, default=73)
    return parser.parse_args()


def _write_run_metadata(
    output_dir: Path,
    spec: LaMannoEvaluationSpec,
    resolved_model_source: str,
    model_source_metadata: dict,
    embedding_config: ev.GenEmbeddingsConfig,
    evaluation_config: ev.EvaluationConfig,
    run_outputs: dict[str, str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "evaluation_name": "lamanno_embedding_evaluation_wrapper",
        "model": asdict(spec),
        "resolved_model_source": resolved_model_source,
        "model_source_metadata": model_source_metadata,
        "annotation_column": ANNOTATION_COLUMN,
        "embedding_config": asdict(embedding_config),
        "evaluation_config": asdict(evaluation_config),
        "outputs": run_outputs,
    }
    with (output_dir / f"{spec.model_label}_evaluation_wrapper_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)


def _extra_cell_annotations_from_adata(adata) -> dict[str, pd.Series]:
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    annotations: dict[str, pd.Series] = {}
    for column in (ANNOTATION_COLUMN, "time"):
        if column in obs.columns:
            annotations[column] = obs[column]
    return annotations


def _timepoint_labels_from_adata(adata, time_column: str = "time") -> list[str]:
    if time_column not in adata.obs.columns:
        raise ValueError(f"AnnData observations are missing `{time_column}`.")
    times = pd.to_numeric(adata.obs[time_column], errors="coerce").dropna().unique()
    return [str(float(value)) for value in sorted(times, key=lambda value: float(value))]


def _precomputed_umaps_from_celltype_group_artifacts(embeddings_dict: dict) -> dict:
    precomputed_umaps: dict = {}
    for model_name, model_data in embeddings_dict.items():
        precomputed_umaps[model_name] = {}
        for dataset_name, dataset_meta in model_data.items():
            group_artifacts = dataset_meta.get("celltype_group_umaps", {})
            precomputed_umaps[model_name][dataset_name] = {}
            for group_name, group_info in group_artifacts.items():
                cells_info = group_info.get("df_cells_umap") or {}
                cells_path = cells_info.get("path")
                if cells_path is None:
                    continue

                centroids = None
                centroids_info = group_info.get("df_celltypes_umap") or {}
                centroids_path = centroids_info.get("path")
                if centroids_path is not None:
                    centroids = pd.read_parquet(centroids_path)

                precomputed_umaps[model_name][dataset_name][group_name] = {
                    "cells": pd.read_parquet(cells_path),
                    "centroids": centroids,
                    "cell_umap_path": cells_path,
                    "centroid_umap_path": centroids_path,
                }
    return precomputed_umaps


def _run_grouped_celltype_umap_plots(
    embeddings_dict: dict,
    adata,
    output_dir: Path,
    random_state: int,
    args: argparse.Namespace,
) -> dict:
    evaluation_config = ev.EvaluationConfig(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.n_components,
        random_state=random_state,
    )
    extra_annotations = _extra_cell_annotations_from_adata(adata)
    # ev.umap_plots(
    #     embeddings_dict=embeddings_dict,
    #     annotation_column=ANNOTATION_COLUMN,
    #     output_dir=str(output_dir / "lineages"),
    #     evaluation_config=evaluation_config,
    #     extra_cell_annotations=extra_annotations,
    #     celltype_groups=LAMANNO_LINEAGES,
    # )
    precomputed_umaps = _precomputed_umaps_from_celltype_group_artifacts(embeddings_dict)
    ev.umap_plots(
        embeddings_dict=embeddings_dict,
        annotation_column=ANNOTATION_COLUMN,
        output_dir=str(output_dir / "lamanno_groups_l"),
        evaluation_config=evaluation_config,
        extra_cell_annotations=extra_annotations,
        celltype_groups=LAMANNO_GROUPS_L,
    )
    # ev.umap_plots(
    #     embeddings_dict=embeddings_dict,
    #     annotation_column=ANNOTATION_COLUMN,
    #     output_dir=str(output_dir / "lamanno_groups_m_split"),
    #     evaluation_config=evaluation_config,
    #     extra_cell_annotations=extra_annotations,
    #     celltype_groups=LAMANNO_GROUPS_M_SPLIT,
    # )
    return precomputed_umaps


def _run_one_spec(spec: LaMannoEvaluationSpec, args: argparse.Namespace) -> dict[str, str]:
    dataset_dir = REVISION_DIR / spec.dataset_id
    evaluation_plot_dir = (
        ANALYSIS_RUN_DIR / spec.dataset_id / spec.model_label / "evaluation_plots"
        if ANALYSIS_RUN_DIR is not None
        else dataset_dir / "evaluation_plots"
    )
    embedding_output_dir = (
        ANALYSIS_RUN_DIR / "embeddings" / spec.model_label
        if ANALYSIS_RUN_DIR is not None
        else dataset_dir
    )
    evaluation_plot_dir.mkdir(parents=True, exist_ok=True)

    model_source = spec.model_id
    model_source_metadata = {"source": "public_huggingface_model", "model_id": model_source}
    print(f"\n=== {spec.model_label}: {model_source} on {spec.dataset_id} ===")
    scrna_test_path = dataset_dir / "datasets" / "scrna_test"
    if not scrna_test_path.exists():
        raise FileNotFoundError(f"Missing local scrna_test at {scrna_test_path}")
    scrna_test = load_from_disk(str(scrna_test_path))
    adata_path = dataset_dir / "adata_test.h5ad"
    if not adata_path.exists():
        raise FileNotFoundError(f"Missing adata_test.h5ad at {adata_path}")
    adata = sc.read_h5ad(adata_path)
    timepoint_labels = _timepoint_labels_from_adata(adata, time_column="time")

    embedding_config = ev.GenEmbeddingsConfig(
        annotation_column=ANNOTATION_COLUMN,
        embedding_models=[model_source],
        model_type="sentence_transformer",
        batch_size=args.batch_size,
        additional_data=timepoint_labels,
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

    outputs: dict[str, str] = {}

    run_umap = not args.only_timepoint_similarity
    run_pseudotime = not args.only_timepoint_similarity and not args.only_umap
    run_timepoint_similarity = not args.only_umap
    run_celltype_similarity = run_pseudotime

    precomputed_umaps = {}
    if run_umap:
        precomputed_umaps = _run_grouped_celltype_umap_plots(
            embeddings_dict=embeddings_dict,
            adata=adata,
            output_dir=evaluation_plot_dir / "embedding_subset_umaps",
            random_state=args.random_state,
            args=args,
        )
        outputs["embedding_subset_umaps_dir"] = str(evaluation_plot_dir / "embedding_subset_umaps")

    if run_pseudotime:
        pseudotime_df = ev.pseudotime(
            embeddings_dict=embeddings_dict,
            eval_data={"adata": adata},
            subfolder_fig_dir=evaluation_plot_dir,
            annotation_column=ANNOTATION_COLUMN,
            evaluation_config=evaluation_config,
            config=ev.PseudotimeConfig(
                lineage=args.lineage,
                cell_origin=args.cell_origin,
                output_dir=evaluation_plot_dir,
            ),
            precomputed_umaps=precomputed_umaps,
        )
        pseudotime_path = evaluation_plot_dir / f"{spec.model_label}_pseudotime.csv"
        pseudotime_df.to_csv(pseudotime_path, index=False)
        outputs["pseudotime_summary"] = str(pseudotime_path)

    if run_celltype_similarity:
        celltype_similarity_df = ev.cell_type_label_similarity(
            embeddings_dict=embeddings_dict,
            annotation_column=ANNOTATION_COLUMN,
            config=ev.CellTypeSimilarityConfig(output_dir=evaluation_plot_dir),
        )
        celltype_similarity_path = evaluation_plot_dir / f"{spec.model_label}_celltype_label_similarity.csv"
        celltype_similarity_df.to_csv(celltype_similarity_path, index=False)
        outputs["celltype_label_similarity"] = str(celltype_similarity_path)

    if run_timepoint_similarity:
        saved_model_name = next(iter(embeddings_dict))
        timepoint_similarity_df = ev.lamanno_timepoint_similarity(
            embeddings_dict=embeddings_dict,
            eval_data={"adata": adata, "adata_path": adata_path},
            config=ev.LaMannoTimepointSimilarityConfig(
                output_dir=evaluation_plot_dir,
                lineages=LAMANNO_LINEAGES,
                lineage_celltype_order=LAMANNO_LINEAGE_CELLTYPE_ORDER,
                annotation_column=ANNOTATION_COLUMN,
                time_column="time",
                model_name_map={saved_model_name: model_source},
            ),
        )
        timepoint_similarity_path = evaluation_plot_dir / f"{spec.model_label}_timepoint_similarity.csv"
        timepoint_similarity_df.to_csv(timepoint_similarity_path, index=False)
        outputs["timepoint_similarity"] = str(timepoint_similarity_path)

    _write_run_metadata(
        output_dir=evaluation_plot_dir,
        spec=spec,
        resolved_model_source=model_source,
        model_source_metadata=model_source_metadata,
        embedding_config=embedding_config,
        evaluation_config=evaluation_config,
        run_outputs=outputs,
    )
    return outputs


def _selected_specs(model_labels: Iterable[str]) -> list[LaMannoEvaluationSpec]:
    return [EVALUATION_SPECS[model_label] for model_label in model_labels]


def main() -> None:
    args = _parse_args()
    all_outputs = {
        spec.model_label: _run_one_spec(spec, args)
        for spec in _selected_specs(args.models)
    }
    print("\nSaved LaManno evaluation outputs:")
    for model_label, outputs in all_outputs.items():
        print(f"- {model_label}")
        for output_name, output_path in outputs.items():
            print(f"  {output_name}: {output_path}")


if __name__ == "__main__":
    main()
