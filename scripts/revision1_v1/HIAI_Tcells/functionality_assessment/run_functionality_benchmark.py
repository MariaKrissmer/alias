from __future__ import annotations

import argparse
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
HIAI_TCELLS_SCRIPT_DIR = PROJECT_ROOT / "scripts" / "revision1_v1" / "HIAI_Tcells"
if str(HIAI_TCELLS_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(HIAI_TCELLS_SCRIPT_DIR))

from alias.evaluation.functionality_benchmark import (  # noqa: E402
    FunctionalityBenchmarkConfig,
    FunctionalityBenchmarkSource,
    run_functionality_benchmark,
)
from util.publication_plotting import ordered_present_models  # noqa: E402


DATASET_ID = "S2_heldout_donor_semantic_200k"
REVISION_DATA_ROOT = Path(
    os.environ.get("ALIAS_REVISION_DATA_ROOT", PROJECT_ROOT / "out" / "data" / "revision1_v1")
)
DATASET_DIR = REVISION_DATA_ROOT / "HIAI_Tcells" / DATASET_ID
EVALUATION_ROOT = Path(
    os.environ.get("HIAI_TCELLS_FUNCTIONALITY_EVALUATION_ROOT", DATASET_DIR / "evaluation_plots")
)
OUTPUT_DIR = Path(
    os.environ.get(
        "HIAI_TCELLS_FUNCTIONALITY_BENCHMARK_OUTPUT_DIR",
        DATASET_DIR / "functionality_assessment" / "benchmark",
    )
)
ALIAS_RESULTS_ROOT = Path(
    os.environ.get(
        "HIAI_TCELLS_FUNCTIONALITY_ALIAS_SCORES_ROOT",
        DATASET_DIR / "functionality_assessment" / "alias_model_scores",
    )
)
FUNCTIONALITY_INPUT_ROOT = Path(
    os.environ.get(
        "HIAI_TCELLS_FUNCTIONALITY_INPUT_ROOT",
        PROJECT_ROOT / "reproducibility" / "inputs" / "hiai_tcells" / "functionality_assessment",
    )
)
FUNCTIONALITY_MAPPING_PATH = FUNCTIONALITY_INPUT_ROOT / "functionality_mapping.csv"
BASELINE_ROOT = FUNCTIONALITY_INPUT_ROOT / "baselines"
EMBEDDING_ROOT = DATASET_DIR / "embeddings"
ANNOTATION_COLUMN = "AIFI_L2"
MAX_CELLS = int(os.environ.get("FUNCTIONALITY_BENCHMARK_MAX_CELLS", "20000"))
BATCH_SIZE = int(os.environ.get("FUNCTIONALITY_BENCHMARK_BATCH_SIZE", "256"))


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


GENERATE_MISSING = _env_bool("FUNCTIONALITY_BENCHMARK_GENERATE_MISSING", default=False)
FORCE_REGENERATE = _env_bool("FUNCTIONALITY_BENCHMARK_FORCE_REGENERATE", default=False)


CELLTYPE_FUNCTIONALITY_DESCRIPTIONS = [
    "Recognize stress ligands not presented on classical MHC molecules.",
    "Recognize lipid, phosphoantigens, and stress ligands via non-conventional pathways.",
    "Produce immunosuppressive cytokines such as IL-10 and TGF-β.",
    "Stably express FOXP3 to maintain suppressive identity.",
    "Recognize bacterial-derived riboflavin metabolites presented on MR1.",
    "Bridge innate and adaptive immunity with semi-invariant TCR and rapid cytokine output.",
    "Retain long-term survival and recall potential.",
    "Rapidly kill target cells upon antigen re-exposure.",
    "Effector memory subset lacks CCR7 and homes to inflamed tissues.",
    "Central memory subset homes to lymphoid tissues and proliferates upon recall.",
    "Serve as unprimed precursors to all helper T cell subsets.",
    "Act as a pool for generating novel responses to previously unseen antigens.",
    "Undergo robust IL-2-driven expansion during acute response.",
    "Increased metabolic activity and cell cycle progression.",
    "Pineapples, mango, blueberry.",
]

FUNCTIONALITY_TRUE_LABEL_MAP = {
    CELLTYPE_FUNCTIONALITY_DESCRIPTIONS[0]: "gamma delta T",
    CELLTYPE_FUNCTIONALITY_DESCRIPTIONS[1]: "gamma delta T",
    CELLTYPE_FUNCTIONALITY_DESCRIPTIONS[2]: "Treg",
    CELLTYPE_FUNCTIONALITY_DESCRIPTIONS[3]: "Treg",
    CELLTYPE_FUNCTIONALITY_DESCRIPTIONS[4]: "MAIT",
    CELLTYPE_FUNCTIONALITY_DESCRIPTIONS[5]: "MAIT",
    CELLTYPE_FUNCTIONALITY_DESCRIPTIONS[6]: "Memory CD8 T cell",
    CELLTYPE_FUNCTIONALITY_DESCRIPTIONS[7]: "Memory CD8 T cell",
    CELLTYPE_FUNCTIONALITY_DESCRIPTIONS[8]: "Memory CD4 T cell",
    CELLTYPE_FUNCTIONALITY_DESCRIPTIONS[9]: "Memory CD4 T cell",
    CELLTYPE_FUNCTIONALITY_DESCRIPTIONS[10]: "Naive CD4 T cell",
    CELLTYPE_FUNCTIONALITY_DESCRIPTIONS[11]: "Naive CD8 T cell",
    CELLTYPE_FUNCTIONALITY_DESCRIPTIONS[12]: "Proliferating T cell",
    CELLTYPE_FUNCTIONALITY_DESCRIPTIONS[13]: "Proliferating T cell",
}


@dataclass(frozen=True)
class ModelSpec:
    label: str
    artifact_name: str
    model_id: str | Path
    fallback_artifact_names: tuple[str, ...] = ()


ABLATION_MODELS = [
    ModelSpec(
        label="MB",
        artifact_name="MB_HIAI_Tcells_S2_N1_200k_lr5e5_e15_epoch_15_ncbi",
        model_id="mariakrissmer/ALIAS-MA",
        fallback_artifact_names=(
            "MB_HIAI_Tcells_S2_N1_200k_lr5e5_e15",
            "MB_HIAI_Tcells_S2_N1_200k_lr5e5",
            "MB_HIAI_Tcells_S2_N1",
        ),
    ),
    ModelSpec(
        label="MJ",
        artifact_name="MJ_HIAI_Tcells_S2_N3_200k_lr5e5",
        model_id="mariakrissmer/ALIAS-MA-star",
    ),
    ModelSpec(
        label="MG",
        artifact_name="MG_HIAI_Tcells_S2_200k_lr5e5_all",
        model_id="mariakrissmer/ALIAS-MB",
        fallback_artifact_names=("MG_HIAI_Tcells_S2_200k_lr5e5", "MG_HIAI_Tcells_S2_200k"),
    ),
    ModelSpec(
        label="MH",
        artifact_name="MH_HIAI_Tcells_S5_200k_lr5e5",
        model_id="mariakrissmer/ALIAS-MC-star",
    ),
    ModelSpec(
        label="MF",
        artifact_name="MF_HIAI_Tcells_S3_200k_lr5e5_all",
        model_id="mariakrissmer/ALIAS-MC",
        fallback_artifact_names=("MF_HIAI_Tcells_S3_200k_lr5e5", "MF_HIAI_Tcells_S3_200k"),
    ),
    ModelSpec(
        label="MI",
        artifact_name="MI_HIAI_Tcells_N1_200k_lr5e5_all",
        model_id="mariakrissmer/ALIAS-MD",
        fallback_artifact_names=("MI_HIAI_Tcells_N1_200k_lr5e5", "MI_HIAI_Tcells_N1_200k"),
    ),
    ModelSpec(
        label="Base",
        artifact_name="Base",
        model_id="neuml/pubmedbert-base-embeddings",
        fallback_artifact_names=("PubMedBERTBase", "pubmedbertbaseembeddings"),
    ),
]

EXTERNAL_BASELINES = [
    FunctionalityBenchmarkSource(
        source_name="cellwhisperer_cell",
        path=BASELINE_ROOT / "cellwhisperer_cell_results.csv",
        source_format="cellwhisperer_results",
        assignment_level="cellwhisperer_cell",
        score_column="mean_auc",
        model_name="CellWhisperer",
    ),
    FunctionalityBenchmarkSource(
        source_name="llama_label",
        path=BASELINE_ROOT / "cell_type_pairwise_wins_llama3_3_70B.csv",
        source_format="llm_matrix",
        assignment_level="llm_label",
        model_name="Llama-3.3-70B",
    ),
    FunctionalityBenchmarkSource(
        source_name="qwen_label",
        path=BASELINE_ROOT / "cell_type_pairwise_wins_qwen3_235B.csv",
        source_format="llm_matrix",
        assignment_level="llm_label",
        model_name="Qwen3-235B",
    ),
]


def _latest_functionality_results(model: ModelSpec) -> Path | None:
    for run_dir in sorted((ALIAS_RESULTS_ROOT / model.label).glob("*"), reverse=True):
        results_path = run_dir / "results_df.csv"
        metadata_path = run_dir / "metadata.json"
        if not results_path.exists() or not metadata_path.exists():
            continue
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("model_id") == str(model.model_id):
            return results_path
    return None


def _generate_functionality_results(model: ModelSpec, *, force_regenerate: bool = False) -> Path:
    from alias.evaluation.functionality_benchmark import generate_alias_functionality_results

    return generate_alias_functionality_results(
        dataset_id=DATASET_ID,
        model_label=model.label,
        model_id=str(model.model_id),
        descriptions=CELLTYPE_FUNCTIONALITY_DESCRIPTIONS,
        true_label_map=FUNCTIONALITY_TRUE_LABEL_MAP,
        embedding_root=EMBEDDING_ROOT,
        evaluation_root=EVALUATION_ROOT,
        results_root=ALIAS_RESULTS_ROOT,
        annotation_column=ANNOTATION_COLUMN,
        max_cells=MAX_CELLS,
        batch_size=BATCH_SIZE,
        force_regenerate=force_regenerate,
    )

def _ablation_sources(
    models: list[ModelSpec],
    *,
    generate_missing: bool = False,
    force_regenerate: bool = False,
) -> list[FunctionalityBenchmarkSource]:
    sources: list[FunctionalityBenchmarkSource] = []
    for model in models:
        if generate_missing and force_regenerate:
            results_path = _generate_functionality_results(
                model,
                force_regenerate=True,
            )
        else:
            results_path = _latest_functionality_results(model)
            if results_path is None:
                if not generate_missing:
                    raise FileNotFoundError(
                        f"No release-model functionality scores for {model.label} ({model.model_id}). "
                        "Run with --generate-missing to score cached cells with the Hub model."
                    )
                results_path = _generate_functionality_results(
                    model,
                    force_regenerate=force_regenerate,
                )
        if results_path is None:
            if not generate_missing:
                print(f"Skipping {model.label}: no functionality_similarity results found.")
                continue
            raise ValueError(f"Could not generate functionality_similarity results for {model.label}.")
        sources.extend(
            [
                FunctionalityBenchmarkSource(
                    source_name="ours_cell",
                    path=results_path,
                    source_format="functionality_similarity",
                    assignment_level="cell",
                    score_column="mean_auc",
                    model_name=model.label,
                ),
                FunctionalityBenchmarkSource(
                    source_name="ours_label",
                    path=results_path,
                    source_format="functionality_similarity",
                    assignment_level="celltype_label",
                    score_column="label_embedding_similarity",
                    model_name=model.label,
                ),
            ]
        )
    return sources


def _select_models(model_names: list[str] | None = None) -> list[ModelSpec]:
    if not model_names:
        return ABLATION_MODELS
    requested = set(model_names)
    selected = [
        model
        for model in ABLATION_MODELS
        if model.label in requested or model.artifact_name in requested
    ]
    if not selected:
        raise ValueError(f"No configured ablation models selected from: {model_names}")
    selected_order = ordered_present_models([model.label for model in selected])
    selected_by_label = {model.label: model for model in selected}
    return [selected_by_label[label] for label in selected_order if label in selected_by_label]


def _available_sources(
    model_names: list[str] | None = None,
    *,
    generate_missing: bool = False,
    force_regenerate: bool = False,
) -> list[FunctionalityBenchmarkSource]:
    selected_models = _select_models(model_names)
    sources = _ablation_sources(
        selected_models,
        generate_missing=generate_missing,
        force_regenerate=force_regenerate,
    )
    for baseline in EXTERNAL_BASELINES:
        if Path(baseline.path).exists():
            sources.append(baseline)
        else:
            print(f"Skipping {baseline.source_name}: missing {baseline.path}")
    if not sources:
        raise ValueError("No functionality benchmark sources were found.")
    return sources


def main(
    model_names: list[str] | None = None,
    *,
    generate_missing: bool = GENERATE_MISSING,
    force_regenerate: bool = FORCE_REGENERATE,
) -> None:
    config = FunctionalityBenchmarkConfig(
        output_dir=OUTPUT_DIR,
        functionality_mapping_path=FUNCTIONALITY_MAPPING_PATH,
        sources=_available_sources(
            model_names,
            generate_missing=generate_missing,
            force_regenerate=force_regenerate,
        ),
        rank_ascending=False,
    )
    outputs = run_functionality_benchmark(config)
    print(f"Saved functionality benchmark to {outputs['run_dir']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HIAI T cell functionality benchmark.")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional model labels or artifact names to include, e.g. MB MG MI.",
    )
    parser.add_argument(
        "--generate-missing",
        action="store_true",
        default=GENERATE_MISSING,
        help="Generate missing functionality_similarity results before benchmarking.",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        default=FORCE_REGENERATE,
        help="Regenerate functionality text embeddings and scores; cached cell embeddings remain required.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        model_names=args.models,
        generate_missing=args.generate_missing,
        force_regenerate=args.force_regenerate,
    )
