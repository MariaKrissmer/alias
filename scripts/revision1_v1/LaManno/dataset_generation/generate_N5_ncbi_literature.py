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


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


SOURCE_H5AD = PROJECT_ROOT / "out" / "data" / "revision1_v1" / "LaManno" / "lamanno_full.h5ad"
BASE_OUTPUT_DIR = SOURCE_H5AD.parent
DATASET_ID = "N5_ncbi_literature"
OUTPUT_DIR = BASE_OUTPUT_DIR / DATASET_ID
RAW_OUTPUT_DIR = BASE_OUTPUT_DIR / "NCBI_raw_collection"
NCBI_QUERY_MODE = os.getenv("NCBI_QUERY_MODE", "exact_title_abstract")
NCBI_TISSUE = os.getenv("NCBI_TISSUE", "Embryo, Mammalian[Mesh]")
RAW_ARTICLES_FILENAME = (
    "lamanno_ncbi_articles_exact_title_abstract_mouse_embryo.csv"
    if NCBI_QUERY_MODE == "exact_title_abstract" and NCBI_TISSUE
    else (
        "lamanno_ncbi_articles_exact_title_abstract.csv"
        if NCBI_QUERY_MODE == "exact_title_abstract"
        else "lamanno_ncbi_articles.csv"
    )
)
RAW_ARTICLES_PATH = Path(
    os.getenv("NCBI_RAW_ARTICLES_PATH", str(RAW_OUTPUT_DIR / RAW_ARTICLES_FILENAME))
)
NCBI_EMAIL = os.getenv("NCBI_EMAIL", "sonia.maria.krissmer@stud.unibas.ch")
RUN_TRIPLETS = _env_bool("NCBI_RUN_TRIPLETS", True)
HF_UPLOAD = _env_bool("NCBI_HF_UPLOAD", True)
REMOVE_MULTILABEL_PMIDS = _env_bool("NCBI_REMOVE_MULTILABEL_PMIDS", True)


def main() -> None:
    adata = sc.read_h5ad(SOURCE_H5AD)
    adata.obs_names_make_unique()

    ncbi_config = DataNCBIConfig(
        dataset_id=DATASET_ID,
        annotation_column="celltype",
        email=NCBI_EMAIL,
        organism="mouse",
        query_mode=NCBI_QUERY_MODE,
        tissue=NCBI_TISSUE,
        max_articles=3000,
        raw_articles_path=RAW_ARTICLES_PATH,
        fetch_if_missing=True,
        output_dir=OUTPUT_DIR,
        save_artifacts=True,
        remove_multilabel_pmids=REMOVE_MULTILABEL_PMIDS,
        collect_cl_terms=False,
        random_state=42,
    )

    dataset_dict, _ = build_datasets(
        adata=adata,
        datasets=["ncbi"],
        ncbi_config=ncbi_config,
    )

    if RUN_TRIPLETS:
        triplet_config = TripletGenerationConfig(
            annotation_column="celltype",
            hard_negative_mining=True,
            loss="MNR",
            testrun=False,
            hf_upload=HF_UPLOAD,
            hf_name=f"LaManno_{DATASET_ID}",
            batch_size=256,
            subset_size=5000,
        )

        train_datasets_ncbi = build_triplets(
            dataset_dict=dataset_dict,
            triplet_config=triplet_config,
        )

        train_datasets_ncbi["ncbi"]["train_MNR_hnm"].save_to_disk(
            str(OUTPUT_DIR / "datasets" / "ncbi_train_MNR_hnm")
        )
        train_datasets_ncbi["ncbi"]["eval_MNR_hnm"].save_to_disk(
            str(OUTPUT_DIR / "datasets" / "ncbi_eval_MNR_hnm")
        )

    del adata
    gc.collect()


if __name__ == "__main__":
    main()
