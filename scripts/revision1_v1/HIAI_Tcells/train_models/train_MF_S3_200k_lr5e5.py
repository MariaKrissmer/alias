from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from alias.model import TrainingSTConfig, train_model
from alias.util.load_hf_model import load_hf_dataset


DATASET_ID = "S3_heldout_donor_nonsemantic_200k"
MODEL_ID = "MF_HIAI_Tcells_S3_200k_lr5e5"
SCRNA_HF_DATASET = f"mariakrissmer/scrna_HIAI_Tcells_{DATASET_ID}"


def main() -> None:
    dataset_dict = {
        "scrna": load_hf_dataset(SCRNA_HF_DATASET),
    }

    train_cfg = TrainingSTConfig(
        model="neuml/pubmedbert-base-embeddings",
        loss="MNR",
        save_to_local=True,
        save_to_hf=True,
        new_model_name=MODEL_ID,
        batch_size=64,
        epochs=8,
        learning_rate=5e-5,
        logging_steps=100,
        save_strategy="epoch",
        load_from_hf=False,
        scrna_hf_dataset=SCRNA_HF_DATASET,
    )

    train_model(
        dataset_dict=dataset_dict,
        datasets=["scrna"],
        train_config=train_cfg,
    )


if __name__ == "__main__":
    main()
