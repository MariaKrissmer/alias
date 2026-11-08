from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_REGISTRY = ROOT / "reproducibility/manifests/model_registry.yaml"
DATASET_REGISTRY = ROOT / "reproducibility/manifests/dataset_registry.yaml"
SELECTED_HIAI_SCRIPTS = {
    "scripts/revision1_v1/HIAI_Tcells/celltype_annotation/_plotting_common.py",
    "scripts/revision1_v1/HIAI_Tcells/celltype_annotation/celltype_annotation.py",
    "scripts/revision1_v1/HIAI_Tcells/celltype_annotation/celltype_annotation_synonym.py",
    "scripts/revision1_v1/HIAI_Tcells/celltype_annotation/plot_ablation_annotation_effect.py",
    "scripts/revision1_v1/HIAI_Tcells/celltype_annotation/plot_annotation_benchmark.py",
    "scripts/revision1_v1/HIAI_Tcells/celltype_annotation/run_celltype_annotation.py",
    "scripts/revision1_v1/HIAI_Tcells/dataset_generation/_scrna_generation_common.py",
    "scripts/revision1_v1/HIAI_Tcells/dataset_generation/generate_N1_N3_ncbi_literature.py",
    "scripts/revision1_v1/HIAI_Tcells/dataset_generation/generate_N4_ncbi_literature_cmv.py",
    "scripts/revision1_v1/HIAI_Tcells/dataset_generation/generate_S2_heldout_donor_semantic_200k.py",
    "scripts/revision1_v1/HIAI_Tcells/dataset_generation/generate_S3_heldout_donor_nonsemantic_200k.py",
    "scripts/revision1_v1/HIAI_Tcells/dataset_generation/generate_S5_heldout_donor_semantic_shuffled_labels_200k.py",
    "scripts/revision1_v1/HIAI_Tcells/dataset_generation/generate_S7_heldout_donor_semantic_cmv_200k.py",
    "scripts/revision1_v1/HIAI_Tcells/functionality_assessment/run_functionality_benchmark.py",
    "scripts/revision1_v1/HIAI_Tcells/functionality_assessment/run_functionality_downstream_cytotoxicity.py",
    "scripts/revision1_v1/HIAI_Tcells/functionality_assessment/run_functionality_downstream_disease_comparison.py",
    "scripts/revision1_v1/HIAI_Tcells/functionality_assessment/run_mait_functionality_synonym_auc.py",
    "scripts/revision1_v1/HIAI_Tcells/model_evaluation/eval_MM_S7_checkpoint_joint_validation_score.py",
    "scripts/revision1_v1/HIAI_Tcells/model_evaluation/eval_checkpoint_joint_validation_score.py",
    "scripts/revision1_v1/HIAI_Tcells/model_evaluation/plot_training_dynamics.py",
    "scripts/revision1_v1/HIAI_Tcells/train_models/train_MB_S2_N1_200k_lr5e5.py",
    "scripts/revision1_v1/HIAI_Tcells/train_models/train_MF_S3_200k_lr5e5.py",
    "scripts/revision1_v1/HIAI_Tcells/train_models/train_MG_S2_200k_lr5e5.py",
    "scripts/revision1_v1/HIAI_Tcells/train_models/train_MH_S5_200k_lr5e5.py",
    "scripts/revision1_v1/HIAI_Tcells/train_models/train_MI_N1_200k_lr5e5.py",
    "scripts/revision1_v1/HIAI_Tcells/train_models/train_MJ_S2_N3_200k_lr5e5.py",
    "scripts/revision1_v1/HIAI_Tcells/train_models/train_MM_S2_N4_200k_lr5e5.py",
    "scripts/revision1_v1/HIAI_Tcells/util/publication_plotting.py",
}
SELECTED_LAMANNO_SCRIPTS = {
    "scripts/revision1_v1/LaManno/dataset_generation/generate_N5_ncbi_literature.py",
    "scripts/revision1_v1/LaManno/dataset_generation/generate_S10_heldout_timepoint_semantic.py",
    "scripts/revision1_v1/LaManno/dataset_generation/generate_S9_heldout_sample_semantic.py",
    "scripts/revision1_v1/LaManno/dataset_generation/heldout_sample_time_eval.py",
    "scripts/revision1_v1/LaManno/evaluation/heldout_timepoint_eval.py",
    "scripts/revision1_v1/LaManno/evaluation/run_lamanno_checkpoint_evaluation.py",
    "scripts/revision1_v1/LaManno/evaluation/run_lamanno_embedding_evaluation.py",
    "scripts/revision1_v1/LaManno/train_models/train_MO_S9_N5_lr5e5.py",
    "scripts/revision1_v1/LaManno/train_models/train_MP_S10_N5_lr5e5.py",
    "scripts/revision1_v1/LaManno/util/__init__.py",
    "scripts/revision1_v1/LaManno/util/checkpoints.py",
    "scripts/revision1_v1/LaManno/util/lineages.py",
}


def test_model_registry_has_expected_canonical_ids_and_scripts() -> None:
    text = MODEL_REGISTRY.read_text(encoding="utf-8")
    for canonical_id in ("MB", "MJ", "MG", "MF", "MH", "MI", "MM", "MO", "MP", "Base"):
        assert f"canonical_id: {canonical_id}" in text
        if canonical_id != "Base":
            assert (ROOT / f"reproducibility/metadata/models/{canonical_id}").is_dir()
    for excluded in ("MO_1", "MQ", "PBMC3k"):
        assert excluded not in text
    for script in re.findall(r"training_script: (.+)", text):
        if script.strip() != "null":
            assert (ROOT / script.strip()).is_file()


def test_dataset_registry_references_existing_metadata_and_scripts() -> None:
    text = DATASET_REGISTRY.read_text(encoding="utf-8")
    expected_dataset_ids = {
        "NCBI_raw_collection",
        "CL_raw_collection",
        "N1_ncbi_literature",
        "N3_ncbi_literature_shuffled_labels",
        "N4_ncbi_literature_cmv",
        "N5_ncbi_literature",
        "S2_heldout_donor_semantic_200k",
        "S3_heldout_donor_nonsemantic_200k",
        "S5_heldout_donor_semantic_shuffled_labels_200k",
        "S7_heldout_donor_semantic_cmv_200k",
        "S9_heldout_sample_semantic",
        "S10_heldout_timepoint_semantic",
    }
    for dataset_id in expected_dataset_ids:
        assert f"  {dataset_id}:" in text
    path_fields = (
        "generation_script",
        "metadata",
        "split_metadata",
        "pmid_manifest",
        "raw_pmid_manifest",
        "query_manifest",
        "raw_cl_terms",
        "cl_terms",
        "n1_query_summary",
        "n3_query_summary",
        "n4_query_summary",
        "n5_query_summary",
        "n1_cl_summary",
        "n3_cl_summary",
        "n4_cl_summary",
    )
    path_pattern = r"(?:" + "|".join(path_fields) + r"): (.+)"
    for path in re.findall(path_pattern, text):
        assert (ROOT / path.strip()).is_file(), path
    assert "n5_cl_summary" not in text
    assert "LaManno/CL_raw_collection" not in text


def test_selected_hiai_scripts_are_exact_and_compile() -> None:
    script_root = ROOT / "scripts/revision1_v1/HIAI_Tcells"
    selected = {
        str(path.relative_to(ROOT))
        for path in script_root.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    assert selected == SELECTED_HIAI_SCRIPTS
    assert (
        "scripts/revision1_v1/HIAI_Tcells/eval_checkpoint_joint_validation_score.py"
        not in selected
    )
    assert (
        "scripts/revision1_v1/HIAI_Tcells/dataset_generation/generate_N1_N2_N3_ncbi_literature.py"
        not in selected
    )
    for script in sorted(SELECTED_HIAI_SCRIPTS):
        path = ROOT / script
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_selected_lamanno_scripts_are_exact_and_compile() -> None:
    script_root = ROOT / "scripts/revision1_v1/LaManno"
    selected = {
        str(path.relative_to(ROOT))
        for path in script_root.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    assert selected == SELECTED_LAMANNO_SCRIPTS
    for excluded in ("MO_1", "MQ", "S11_heldout_sample_semantic_no_time"):
        for script in selected:
            assert excluded not in (ROOT / script).read_text(encoding="utf-8")
    for script in sorted(SELECTED_LAMANNO_SCRIPTS):
        path = ROOT / script
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_manifests_are_portable_and_complete() -> None:
    manifest_dir = ROOT / "reproducibility/manifests"
    for filename in (
        "query_manifest.csv",
        "pmid_manifest.csv",
        "ncbi_raw_pmid_manifest.csv",
        "split_manifest.json",
    ):
        assert (manifest_dir / filename).is_file()
    assert (
        ROOT
        / "reproducibility/metadata/datasets/HIAI_Tcells/CL_raw_collection/hiai_tcells_cl_terms.csv"
    ).is_file()
    assert not (
        ROOT
        / "reproducibility/metadata/datasets/LaManno/CL_raw_collection/records/N5_cl_sctype_term_summary.csv"
    ).exists()
    assert not (
        ROOT
        / "reproducibility/metadata/datasets/LaManno/CL_raw_collection/lamanno_cl_terms.csv"
    ).exists()
    assert "N2_ncbi_literature_heldout_celltype" not in (
        ROOT
        / "scripts/revision1_v1/HIAI_Tcells/dataset_generation/generate_N1_N3_ncbi_literature.py"
    ).read_text(encoding="utf-8")
    forbidden = re.compile(r"/Users/|/home/|/scratch/|/Volumes/|/mnt/|alias-private")
    for path in (ROOT / "reproducibility").rglob("*"):
        if path.is_file() and path.suffix in {".csv", ".json", ".yaml", ".md", ".py"}:
            assert not forbidden.search(path.read_text(encoding="utf-8")), path


def test_no_private_or_generated_artifacts_are_selected() -> None:
    forbidden_dirs = {"logs", "__pycache__", "checkpoints"}
    for base in (ROOT / "scripts", ROOT / "reproducibility"):
        for path in base.rglob("*"):
            assert not (forbidden_dirs & set(path.parts)), path
            assert path.name not in {".DS_Store"}
            assert path.suffix != ".log"
