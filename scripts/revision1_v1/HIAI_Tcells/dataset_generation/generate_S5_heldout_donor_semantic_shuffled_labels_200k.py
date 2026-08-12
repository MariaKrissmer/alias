from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _scrna_generation_common import (
    BASE_OUTPUT_DIR,
    DONOR_TEST_SIZE,
    RANDOM_STATE,
    SOURCE_H5AD,
    TEST_SIZE,
    TOTAL_CELLS,
    run_scrna_generation,
)

from alias.data import DatascRNAConfig


DATASET_ID = "S5_heldout_donor_semantic_shuffled_labels_200k"
OUTPUT_DIR = BASE_OUTPUT_DIR / DATASET_ID


def main() -> None:
    scrna_config = DatascRNAConfig(
        dataset_id=DATASET_ID,
        annotation_column="AIFI_L2",
        donor_column="subject.subjectGuid",
        split_strategy="heldout_donor",
        donor_test_size=DONOR_TEST_SIZE,
        total_cells=TOTAL_CELLS,
        output_dir=OUTPUT_DIR,
        source=str(SOURCE_H5AD),
        save_artifacts=True,
        preprocessing=True,
        highly_variable_genes=True,
        housekeeping_genes=True,
        cs_length=[50],
        semantic=True,
        shuffle_train_labels=True,
        label_shuffle_seed=RANDOM_STATE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    run_scrna_generation(
        scrna_config=scrna_config,
        train_semantic=True,
        test_semantic=False,
        hf_name=f"HIAI_Tcells_{DATASET_ID}",
    )


if __name__ == "__main__":
    main()
