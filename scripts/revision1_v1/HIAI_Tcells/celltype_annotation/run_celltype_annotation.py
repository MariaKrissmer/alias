from __future__ import annotations

import argparse
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
    return parser.parse_args()


def main() -> None:
    args = _args()
    model_names = _parse_model_names(args.models)

    if args.force_normal:
        os.environ["HIAI_TCELLS_ANNOTATION_FORCE"] = "1"
    if args.force_synonym:
        os.environ["HIAI_TCELLS_SYNONYM_FORCE"] = "1"
    os.environ["HIAI_TCELLS_SYNONYM_LABEL_PLOTS"] = "1"
    os.environ["HIAI_TCELLS_SYNONYM_LABEL_MODELS"] = args.synonym_label_plot_models

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
