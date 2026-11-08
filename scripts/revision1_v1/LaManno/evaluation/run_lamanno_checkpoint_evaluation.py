from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import runpy
from typing import Iterator


def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src" / "alias").exists():
            return parent
    raise RuntimeError("Could not resolve project root from this script location.")


PROJECT_ROOT = _find_project_root()
REVISION_DIR = PROJECT_ROOT / "out" / "data" / "revision1_v1" / "LaManno"
EVALUATION_DATASET_ID = "S9_heldout_sample_semantic"
NCBI_DATASET_ID = "N5_ncbi_literature"
CHECKPOINT_VALIDATION_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "revision1_v1"
    / "HIAI_Tcells"
    / "model_evaluation"
    / "eval_checkpoint_joint_validation_score.py"
)

MODEL_CONFIGS = {
    "MO": {
        "model_id": "MO_LaManno_S9_N5_lr5e5",
        "model_root_env": "MO_LAMANNO_MODEL_ROOT",
        "default_model_root": PROJECT_ROOT / "models" / "MO_LaManno_S9_N5_lr5e5_all",
    },
    "MP": {
        "model_id": "MP_LaManno_S10_N5_lr5e5",
        "model_root_env": "MP_LAMANNO_MODEL_ROOT",
        "default_model_root": PROJECT_ROOT / "models" / "MP_LaManno_S10_N5_lr5e5_all",
    },
}
DEFAULT_MODEL_LABELS = ("MO", "MP")

SCRNA_EVAL_PATH = (
    REVISION_DIR
    / EVALUATION_DATASET_ID
    / "datasets"
    / "scrna_eval_MNR_hnm"
)
NCBI_EVAL_PATH = (
    REVISION_DIR
    / NCBI_DATASET_ID
    / "datasets"
    / "ncbi_eval_MNR_hnm"
)
OUTPUT_ROOT = (
    REVISION_DIR
    / EVALUATION_DATASET_ID
    / "checkpoint_validation_score"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run HIAI-style checkpoint validation for LaManno MO/MP checkpoints "
            "using the S9 scRNA eval split and N5 NCBI eval split."
        )
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODEL_LABELS),
        choices=sorted(MODEL_CONFIGS),
        help="LaManno checkpoint model families to evaluate.",
    )
    parser.add_argument(
        "--checkpoints",
        default=os.environ.get("LAMANNO_CHECKPOINTS", ""),
        help=(
            "Comma-separated checkpoint folder names. Defaults to all checkpoint-* "
            "and epoch_* folders discovered by the HIAI checkpoint validation script."
        ),
    )
    parser.add_argument(
        "--max-eval-triplets",
        default=os.environ.get("LAMANNO_MAX_EVAL_TRIPLETS", "1000"),
        help="Maximum eval triplets per dataset. Use 0 to evaluate all triplets.",
    )
    parser.add_argument(
        "--batch-size",
        default=os.environ.get("LAMANNO_CHECKPOINT_BATCH_SIZE", "256"),
        help="SentenceTransformer encode batch size.",
    )
    parser.add_argument(
        "--force-regenerate-embedding-cache",
        action="store_true",
        help="Regenerate cached triplet embeddings for checkpoint validation.",
    )
    return parser.parse_args()


@contextmanager
def _temporary_environment(updates: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _model_environment(model_key: str, args: argparse.Namespace) -> dict[str, str]:
    model_config = MODEL_CONFIGS[model_key]
    model_id = model_config["model_id"]
    model_root = Path(
        os.environ.get(
            model_config["model_root_env"],
            str(model_config["default_model_root"]),
        )
    )
    output_dir = OUTPUT_ROOT / model_id

    env = {
        "JOINT_VALIDATION_MODEL_ID": model_id,
        "JOINT_VALIDATION_MODEL_ROOT": str(model_root),
        "JOINT_VALIDATION_SCRNA_EVAL_PATH": str(SCRNA_EVAL_PATH),
        "JOINT_VALIDATION_NCBI_EVAL_PATH": str(NCBI_EVAL_PATH),
        "JOINT_VALIDATION_OUTPUT_DIR": str(output_dir),
        "JOINT_VALIDATION_EMBEDDING_CACHE_DIR": str(output_dir / "embedding_cache"),
        "JOINT_VALIDATION_MAX_EVAL_TRIPLETS": str(args.max_eval_triplets),
        "JOINT_VALIDATION_BATCH_SIZE": str(args.batch_size),
        "JOINT_VALIDATION_CHECKPOINTS": args.checkpoints,
    }
    if args.force_regenerate_embedding_cache:
        env["JOINT_VALIDATION_FORCE_REGENERATE_EMBEDDING_CACHE"] = "true"
    return env


def main() -> None:
    args = _parse_args()
    for model_key in args.models:
        model_env = _model_environment(model_key, args)
        print(
            "\nRunning LaManno checkpoint validation "
            f"for {model_key} on {EVALUATION_DATASET_ID} eval split."
        )
        print(f"Model root: {model_env['JOINT_VALIDATION_MODEL_ROOT']}")
        print(f"scRNA eval: {SCRNA_EVAL_PATH}")
        print(f"NCBI eval: {NCBI_EVAL_PATH}")
        print(f"Output: {model_env['JOINT_VALIDATION_OUTPUT_DIR']}")
        with _temporary_environment(model_env):
            runpy.run_path(str(CHECKPOINT_VALIDATION_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
