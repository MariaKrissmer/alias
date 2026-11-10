from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "revision1_v1"
    / "HIAI_Tcells"
    / "functionality_assessment"
    / "run_mait_functionality_synonym_auc.py"
)


def _load_runner_module():
    spec = importlib.util.spec_from_file_location("hiai_tcells_mait_functionality_synonym_auc", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_mait_description_variants_include_true_and_paraphrases() -> None:
    module = _load_runner_module()

    variants = module.mait_description_variants()

    assert list(variants.columns) == [
        "description_id",
        "variant_type",
        "source_description",
        "description_text",
        "target_cell_type",
    ]
    assert set(variants["variant_type"]) == {"true", "synonym"}
    assert variants["target_cell_type"].unique().tolist() == ["MAIT"]
    assert len(variants[variants["variant_type"] == "true"]) == 2
    assert len(variants[variants["variant_type"] == "synonym"]) == 6
    assert variants["description_id"].is_unique


def test_summarize_mait_auc_groups_true_and_synonym_variants() -> None:
    module = _load_runner_module()
    long_df = pd.DataFrame(
        {
            "model": ["MB", "MB", "MB", "MB", "MI", "MI"],
            "artifact_name": ["MB_artifact"] * 4 + ["MI_artifact"] * 2,
            "description_id": ["true_1", "true_2", "synonym_1", "synonym_2", "true_1", "synonym_1"],
            "variant_type": ["true", "true", "synonym", "synonym", "true", "synonym"],
            "description_text": ["a", "b", "c", "d", "a", "c"],
            "target_cell_type": ["MAIT"] * 6,
            "roc_auc": [0.8, 0.6, 0.7, 0.9, 0.5, 0.75],
        }
    )

    summary = module.summarize_mait_auc(long_df)

    mb_true = summary[
        (summary["model"] == "MB") & (summary["variant_type"] == "true")
    ].iloc[0]
    mb_synonym = summary[
        (summary["model"] == "MB") & (summary["variant_type"] == "synonym")
    ].iloc[0]
    assert mb_true["mean_roc_auc"] == 0.7
    assert mb_true["n_descriptions"] == 2
    assert mb_synonym["mean_roc_auc"] == 0.8
    assert mb_synonym["n_descriptions"] == 2
    assert list(summary.columns) == [
        "model",
        "artifact_name",
        "variant_type",
        "mean_roc_auc",
        "sd_roc_auc",
        "n_descriptions",
    ]


