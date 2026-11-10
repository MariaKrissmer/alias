from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


DEFAULT_MODELS = [
    "CellTypist",
    "SingleR",
    "Base",
    "MI",
    "MF",
    "MG",
    "MB",
    "MJ",
    "MH",
]


def _parse_model_names(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return DEFAULT_MODELS
    return [name.strip() for name in value.split(",") if name.strip()]


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run HIAI T-cell cell-type annotation evaluations and regenerate "
            "the joint-format plots."
        )
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help=(
            "Comma-separated model labels to evaluate. Use labels such as "
            "CellTypist,SingleR,Base,MI,MF,MG,MB,MJ,MH,ML, or all."
        ),
    )
    parser.add_argument(
        "--skip-normal",
        action="store_true",
        help="Skip the normal annotation benchmark.",
    )
    parser.add_argument(
        "--skip-synonym",
        action="store_true",
        help="Skip the synonym annotation benchmark.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip joint-format plot regeneration.",
    )
    parser.add_argument(
        "--force-normal",
        action="store_true",
        help="Force regeneration of normal annotation predictions and embeddings.",
    )
    parser.add_argument(
        "--force-synonym",
        action="store_true",
        help="Force regeneration of synonym annotation predictions and embeddings.",
    )
    parser.add_argument(
        "--synonym-label-plot-models",
        default="MB",
        help=(
            "Comma-separated model labels for per-synonym UMAP/hist/ROC-AUC plots. "
            "Defaults to MB."
        ),
    )
    parser.add_argument(
        "--label-umaps-only",
        action="store_true",
        help="Generate only the cell type label UMAPs for --models.",
    )
    parser.add_argument(
        "--figure2c-embeddings",
        type=Path,
        help="Recreate Figure 2C from the saved embedding manifest in one analysis run.",
    )
    parser.add_argument(
        "--figure2d-embeddings",
        type=Path,
        help="Recreate the Figure 2D Treg/FOXP3+ synonym plots from saved embeddings.",
    )
    return parser.parse_args()


def load_figure2c_embedding_manifest(path: Path, model_names: list[str]) -> dict:
    """Validate that every Figure 2C model uses artifacts from its analysis run."""
    import celltype_annotation

    from alias.evaluation.embedding import clean_model_name
    from alias.util.plots.color_definition import PUBLICATION_ABLATION_MODEL_ORDER

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("dataset_id") != celltype_annotation.DATASET_ID:
        raise ValueError("Figure 2C embedding manifest has the wrong dataset ID.")
    run_dir = Path(payload["analysis_run"]).resolve()
    if not Path(path).resolve().is_relative_to(run_dir):
        raise ValueError("Figure 2C embedding manifest is outside its analysis run.")
    if not payload.get("observed_labels"):
        raise ValueError("Figure 2C embedding manifest has no observed cell-type labels.")
    model_configs = {
        config["name"]: config
        for config in celltype_annotation.COMPARISON_MODELS
        if config["kind"] == "sentence_transformer_similarity"
        and config["name"] in PUBLICATION_ABLATION_MODEL_ORDER
    }
    for internal_name in model_names:
        if internal_name not in model_configs:
            raise ValueError(f"Figure 2C requires a published ablation model: {internal_name}")
        model_id = str(model_configs[internal_name]["model"])
        if payload.get("model_sources", {}).get(internal_name) != model_id:
            raise ValueError(f"Wrong Hugging Face model source for {internal_name}: {model_id}")
        model_key = clean_model_name(model_id)
        dataset_artifacts = payload["embeddings"][model_key]["scrna"]
        for artifact_name in ("df_cells", "df_celltypes"):
            artifact_path = Path(dataset_artifacts[artifact_name]["path"]).resolve()
            if not artifact_path.is_relative_to(run_dir / "embeddings"):
                raise ValueError(
                    f"{internal_name} {artifact_name} is outside the Figure 2 analysis run: "
                    f"{artifact_path}"
                )
            if not artifact_path.is_file():
                raise FileNotFoundError(f"Missing Figure 2 embedding artifact: {artifact_path}")
    return payload


def main() -> None:
    args = _args()
    model_names = _parse_model_names(args.models)

    if args.figure2d_embeddings is not None:
        import celltype_annotation_synonym

        if model_names != ["MB"]:
            raise ValueError("Figure 2D requires internal model MB (public ALIAS-MA).")
        manifest = load_figure2c_embedding_manifest(args.figure2d_embeddings, model_names)
        run_dir = Path(manifest["analysis_run"]).resolve()
        output_root = Path(os.environ["HIAI_TCELLS_SYNONYM_OUTPUT_DIR"]).resolve()
        if not output_root.is_relative_to(run_dir):
            raise ValueError(f"Figure 2D output must be inside {run_dir}")
        if Path(os.environ["HIAI_TCELLS_EMBEDDING_ROOT"]).resolve() != run_dir / "embeddings":
            raise ValueError("HIAI_TCELLS_EMBEDDING_ROOT must be the Figure 2 run's embeddings folder.")
        celltype_annotation_synonym.run_selected_synonym_label_similarity_from_embeddings(
            embeddings_dict=manifest["embeddings"],
            observed_labels=set(map(str, manifest["observed_labels"])),
            model_name="MB",
            canonical_label="Treg",
            synonym_label="FOXP3+ T cell",
            output_root=output_root,
        )
        return

    if args.figure2c_embeddings is not None:
        import celltype_annotation
        import celltype_annotation_synonym
        import plot_ablation_annotation_effect

        manifest = load_figure2c_embedding_manifest(args.figure2c_embeddings, model_names)
        run_dir = Path(manifest["analysis_run"]).resolve()
        required_output_roots = (
            "HIAI_TCELLS_ANNOTATION_OUTPUT_DIR",
            "HIAI_TCELLS_SYNONYM_OUTPUT_DIR",
        )
        for variable in required_output_roots:
            output_path = Path(os.environ[variable]).resolve()
            if not output_path.is_relative_to(run_dir):
                raise ValueError(f"{variable} must be inside {run_dir}")
        if Path(os.environ["HIAI_TCELLS_EMBEDDING_ROOT"]).resolve() != run_dir / "embeddings":
            raise ValueError("HIAI_TCELLS_EMBEDDING_ROOT must be the Figure 2 run's embeddings folder.")

        celltype_annotation.run(
            model_names=model_names,
            embeddings_dict=manifest["embeddings"],
        )
        plot_ablation_annotation_effect.main(model_names=model_names)
        celltype_annotation_synonym.run_balanced_accuracy_from_embeddings(
            embeddings_dict=manifest["embeddings"],
            model_names=model_names,
            observed_labels=set(map(str, manifest["observed_labels"])),
        )
        return

    if args.force_normal:
        os.environ["HIAI_TCELLS_ANNOTATION_FORCE"] = "1"
    if args.force_synonym:
        os.environ["HIAI_TCELLS_SYNONYM_FORCE"] = "1"
    os.environ["HIAI_TCELLS_SYNONYM_LABEL_PLOTS"] = "1"
    os.environ["HIAI_TCELLS_SYNONYM_LABEL_MODELS"] = args.synonym_label_plot_models

    if args.label_umaps_only:
        import celltype_annotation_synonym

        celltype_annotation_synonym.run_label_umaps(model_names=model_names)
        return

    if not args.skip_normal:
        import celltype_annotation

        normal_models = model_names
        print(f"Running normal annotation models: {normal_models}")
        celltype_annotation.run(model_names=normal_models)

    if not args.skip_synonym:
        import celltype_annotation_synonym

        synonym_models = [
            model
            for model in model_names
            if model not in {"CellTypist", "SingleR"}
        ]
        print(f"Running synonym annotation models: {synonym_models}")
        celltype_annotation_synonym.run(model_names=synonym_models)

    if not args.skip_plots:
        import plot_ablation_annotation_effect
        import plot_annotation_benchmark

        print("Regenerating joint-format annotation benchmark plots.")
        plot_annotation_benchmark.main(model_names=model_names)
        print("Regenerating joint-format ablation annotation plots.")
        plot_ablation_annotation_effect.main(model_names=model_names)


if __name__ == "__main__":
    main()
