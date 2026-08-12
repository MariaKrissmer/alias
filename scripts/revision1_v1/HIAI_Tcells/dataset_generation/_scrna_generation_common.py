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

from alias.data import DatascRNAConfig, TripletGenerationConfig, build_datasets, build_triplets
from alias.util.hiai_subsets import DEFAULT_HIAI_T_CELL_L2, subset_hiai_t_cells


SOURCE_H5AD = (
    PROJECT_ROOT
    / "out"
    / "data"
    / "revision1_v1"
    / "HIAI"
    / "human_immune_health_atlas_full.h5ad"
)
BASE_OUTPUT_DIR = PROJECT_ROOT / "out" / "data" / "revision1_v1" / "HIAI_Tcells"

TOTAL_CELLS = 200_000
TEST_SIZE = 0.50
DONOR_TEST_SIZE = 0.50
RANDOM_STATE = 42


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def load_hiai_tcell_subset():
    adata = sc.read_h5ad(SOURCE_H5AD)
    adata_subset = subset_hiai_t_cells(
        adata,
        annotation_column="AIFI_L2",
        t_cell_labels=DEFAULT_HIAI_T_CELL_L2,
        min_keep=2000,
        fraction=0.2,
        random_state=RANDOM_STATE,
        use_raw=True,
        normalize_gdt=True,
    )
    adata_subset.obs_names_make_unique()
    del adata
    gc.collect()
    return adata_subset


def run_scrna_generation(
    *,
    scrna_config: DatascRNAConfig,
    train_semantic: bool,
    test_semantic: bool = False,
    hf_name: str,
) -> None:
    adata = load_hiai_tcell_subset()
    dataset_dict, _ = build_datasets(
        adata=adata,
        datasets=["scrna"],
        scrna_config=scrna_config,
        train_semantic=train_semantic,
        test_semantic=test_semantic,
    )
    del adata
    gc.collect()

    if not env_bool("SCRNA_RUN_TRIPLETS", True):
        print("SCRNA_RUN_TRIPLETS is disabled; skipping triplet generation.")
        return

    output_dir = Path(scrna_config.output_dir)
    triplet_config = TripletGenerationConfig(
        annotation_column="AIFI_L2",
        hard_negative_mining=True,
        loss="MNR",
        testrun=False,
        hf_upload=env_bool("SCRNA_HF_UPLOAD", True),
        hf_name=hf_name,
        batch_size=256,
        subset_size=5000,
    )

    train_datasets_scrna = build_triplets(
        dataset_dict=dataset_dict,
        triplet_config=triplet_config,
    )

    train_datasets_scrna["scrna"]["train_MNR_hnm"].save_to_disk(
        str(output_dir / "datasets" / "scrna_train_MNR_hnm")
    )
    train_datasets_scrna["scrna"]["eval_MNR_hnm"].save_to_disk(
        str(output_dir / "datasets" / "scrna_eval_MNR_hnm")
    )
