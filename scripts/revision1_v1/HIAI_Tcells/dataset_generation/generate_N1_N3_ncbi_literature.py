from __future__ import annotations

import gc
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import scanpy as sc

from alias.data import (
    DataNCBIConfig,
    TripletGenerationConfig,
    build_datasets,
    build_triplets,
)
from alias.util.hiai_subsets import DEFAULT_HIAI_T_CELL_L2, subset_hiai_t_cells


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


SOURCE_H5AD = (
    PROJECT_ROOT
    / "out"
    / "data"
    / "revision1_v1"
    / "HIAI"
    / "human_immune_health_atlas_full.h5ad"
)
BASE_OUTPUT_DIR = PROJECT_ROOT / "out" / "data" / "revision1_v1" / "HIAI_Tcells"
RAW_OUTPUT_DIR = BASE_OUTPUT_DIR / "NCBI_raw_collection"
CL_RAW_OUTPUT_DIR = BASE_OUTPUT_DIR / "CL_raw_collection"
REPORTS_DIR = BASE_OUTPUT_DIR / "reports"
NCBI_QUERY_MODE = os.getenv("NCBI_QUERY_MODE", "exact_title_abstract")
NCBI_TISSUE = os.getenv("NCBI_TISSUE", "Blood Cells[Mesh]")
RAW_ARTICLES_FILENAME = (
    "hiai_tcells_ncbi_articles_exact_title_abstract_blood_cells.csv"
    if NCBI_QUERY_MODE == "exact_title_abstract" and NCBI_TISSUE
    else (
        "hiai_tcells_ncbi_articles_exact_title_abstract.csv"
        if NCBI_QUERY_MODE == "exact_title_abstract"
        else "hiai_tcells_ncbi_articles.csv"
    )
)
RAW_ARTICLES_PATH = Path(
    os.getenv("NCBI_RAW_ARTICLES_PATH", str(RAW_OUTPUT_DIR / RAW_ARTICLES_FILENAME))
)
CL_DESCRIPTION_PATH = Path(
    os.getenv(
        "CL_DESCRIPTION_PATH",
        str(
            PROJECT_ROOT
            / "out"
            / "data"
            / "revision1_v1"
            / "HIAI"
            / "celltype_descriptions"
            / "hiai_aifi_l2_descriptions_for_evaluation.csv"
        ),
    )
)
CL_RAW_PATH = Path(
    os.getenv(
        "CL_RAW_PATH",
        str(CL_RAW_OUTPUT_DIR / "hiai_tcells_cl_terms.csv"),
    )
)
NCBI_EMAIL = os.getenv("NCBI_EMAIL")
RUN_TRIPLETS = _env_bool("NCBI_RUN_TRIPLETS", True)
HF_UPLOAD = _env_bool("NCBI_HF_UPLOAD", True)
SAVE_TCELL_SUBSET = _env_bool("HIAI_TCELL_SAVE_SUBSET", True)
COLLECT_CL_TERMS = _env_bool("COLLECT_CL_TERMS", True)
REMOVE_MULTILABEL_PMIDS = _env_bool("NCBI_REMOVE_MULTILABEL_PMIDS", True)
CL_INFUSION_MODE = os.getenv("CL_INFUSION_MODE", "title_abstract")


def _save_subset_reports(adata) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    counts = (
        adata.obs["AIFI_L2"]
        .value_counts()
        .rename_axis("cell_type")
        .reset_index(name="n_cells")
    )
    counts.to_csv(REPORTS_DIR / "tcells_subset_celltype_counts.csv", index=False)

    if SAVE_TCELL_SUBSET:
        adata.write_h5ad(BASE_OUTPUT_DIR / "hiai_tcells_subset.h5ad")


def _build_ncbi_variant(
    *,
    adata,
    dataset_id: str,
    heldout_values: list[str] | None = None,
    shuffle_labels: bool = False,
) -> None:
    if not NCBI_EMAIL:
        raise RuntimeError(
            "Set NCBI_EMAIL to an email address accepted by NCBI before running "
            "the literature retrieval workflow."
        )
    output_dir = BASE_OUTPUT_DIR / dataset_id
    ncbi_config = DataNCBIConfig(
        dataset_id=dataset_id,
        annotation_column="AIFI_L2",
        email=NCBI_EMAIL,
        organism="homo sapiens",
        query_mode=NCBI_QUERY_MODE,
        tissue=NCBI_TISSUE,
        max_articles=3000,
        raw_articles_path=RAW_ARTICLES_PATH,
        fetch_if_missing=True,
        output_dir=output_dir,
        save_artifacts=True,
        heldout_values=heldout_values,
        heldout_key="cell_types",
        remove_multilabel_pmids=REMOVE_MULTILABEL_PMIDS,
        shuffle_labels=shuffle_labels,
        label_shuffle_seed=42,
        collect_cl_terms=COLLECT_CL_TERMS,
        cl_description_path=CL_DESCRIPTION_PATH,
        cl_raw_path=CL_RAW_PATH,
        cl_split_descriptions=True,
        cl_min_sentence_words=8,
        cl_infusion_mode=CL_INFUSION_MODE,
        random_state=42,
    )

    dataset_dict, _ = build_datasets(
        adata=adata,
        datasets=["ncbi"],
        ncbi_config=ncbi_config,
    )

    if not RUN_TRIPLETS:
        return

    triplet_config = TripletGenerationConfig(
        annotation_column="AIFI_L2",
        hard_negative_mining=True,
        loss="MNR",
        testrun=False,
        hf_upload=HF_UPLOAD,
        hf_name=f"HIAI_Tcells_{dataset_id}",
        batch_size=256,
        subset_size=5000,
    )

    train_datasets_ncbi = build_triplets(
        dataset_dict=dataset_dict,
        triplet_config=triplet_config,
    )

    train_datasets_ncbi["ncbi"]["train_MNR_hnm"].save_to_disk(
        str(output_dir / "datasets" / "ncbi_train_MNR_hnm")
    )
    train_datasets_ncbi["ncbi"]["eval_MNR_hnm"].save_to_disk(
        str(output_dir / "datasets" / "ncbi_eval_MNR_hnm")
    )


def main() -> None:
    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(SOURCE_H5AD)
    adata.obs_names_make_unique()
    adata_subset = subset_hiai_t_cells(
        adata,
        annotation_column="AIFI_L2",
        t_cell_labels=DEFAULT_HIAI_T_CELL_L2,
        min_keep=2000,
        fraction=0.2,
        random_state=42,
        use_raw=True,
    )
    del adata
    gc.collect()

    _save_subset_reports(adata_subset)

    _build_ncbi_variant(
        adata=adata_subset,
        dataset_id="N1_ncbi_literature",
    )
    _build_ncbi_variant(
        adata=adata_subset,
        dataset_id="N3_ncbi_literature_shuffled_labels",
        shuffle_labels=True,
    )

    del adata_subset
    gc.collect()


if __name__ == "__main__":
    main()
