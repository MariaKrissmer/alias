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
    DatascRNAConfig,
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
DATASET_ID = "S9_heldout_sample_semantic"
OUTPUT_DIR = SOURCE_H5AD.parent / DATASET_ID
RUN_TRIPLETS = _env_bool("SCRNA_RUN_TRIPLETS", True)
HF_UPLOAD = _env_bool("SCRNA_HF_UPLOAD", True)


def main() -> None:
    adata = sc.read_h5ad(SOURCE_H5AD)
    adata.obs_names_make_unique()

    scrna_config = DatascRNAConfig(
        dataset_id=DATASET_ID,
        annotation_column="celltype",
        split_strategy="heldout_group",
        group_column="sample",
        group_key="samples",
        group_test_size=0.40,
        time_column="time",
        output_dir=OUTPUT_DIR,
        source=str(SOURCE_H5AD),
        save_artifacts=True,
        preprocessing=True,
        highly_variable_genes=True,
        housekeeping_genes=True,
        cs_length=[50],
        semantic=True,
        test_size=0.40,
        stratified_subsample=False,
        random_state=42,
    )

    dataset_dict, _ = build_datasets(
        adata=adata,
        datasets=["scrna"],
        scrna_config=scrna_config,
        train_semantic=True,
        test_semantic=False,
    )
    del adata
    gc.collect()

    if not RUN_TRIPLETS:
        return

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

    train_datasets_scrna = build_triplets(
        dataset_dict=dataset_dict,
        triplet_config=triplet_config,
    )

    train_datasets_scrna["scrna"]["train_MNR_hnm"].save_to_disk(
        str(OUTPUT_DIR / "datasets" / "scrna_train_MNR_hnm")
    )
    train_datasets_scrna["scrna"]["eval_MNR_hnm"].save_to_disk(
        str(OUTPUT_DIR / "datasets" / "scrna_eval_MNR_hnm")
    )


if __name__ == "__main__":
    main()
