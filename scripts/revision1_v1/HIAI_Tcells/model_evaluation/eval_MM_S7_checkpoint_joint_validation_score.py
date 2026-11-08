from __future__ import annotations

import os
import runpy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATASET_ID = "S7_heldout_donor_semantic_cmv_200k"
MODEL_ID = "MM_HIAI_Tcells_S2_N4_200k_lr5e5"


def _set_default_env() -> None:
    os.environ.setdefault("JOINT_VALIDATION_MODEL_ID", MODEL_ID)
    os.environ.setdefault(
        "JOINT_VALIDATION_MODEL_ROOT",
        str(PROJECT_ROOT / "models" / f"{MODEL_ID}_all"),
    )
    os.environ.setdefault(
        "JOINT_VALIDATION_SCRNA_EVAL_PATH",
        str(
            PROJECT_ROOT
            / "out"
            / "data"
            / "revision1_v1"
            / "HIAI_Tcells"
            / DATASET_ID
            / "datasets"
            / "scrna_eval_MNR_hnm"
        ),
    )
    os.environ.setdefault(
        "JOINT_VALIDATION_NCBI_EVAL_PATH",
        str(
            PROJECT_ROOT
            / "out"
            / "data"
            / "revision1_v1"
            / "HIAI_Tcells"
            / "N4_ncbi_literature_cmv"
            / "datasets"
            / "ncbi_eval_MNR_hnm"
        ),
    )
    os.environ.setdefault(
        "JOINT_VALIDATION_OUTPUT_DIR",
        str(
            PROJECT_ROOT
            / "out"
            / "data"
            / "revision1_v1"
            / "HIAI_Tcells"
            / DATASET_ID
            / "model_evaluation"
            / "checkpoint_joint_validation_score"
            / MODEL_ID
        ),
    )


def main() -> None:
    _set_default_env()
    runpy.run_path(
        str(Path(__file__).resolve().parent / "eval_checkpoint_joint_validation_score.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
