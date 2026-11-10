from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import scanpy as sc
from datasets import load_from_disk

from alias.data import (
    DatascRNAConfig,
    TripletGenerationConfig,
    build_datasets,
    build_triplets,
)
from alias.model import TrainingSTConfig, train_model


STUDY_ID = "pbmc3k_cell_sentence_length_stability"
STUDY_ROOT = Path(
    os.environ.get(
        "PBMC3K_STUDY_ROOT", PROJECT_ROOT / "out" / "data" / "revision1_v1" / "pbmc3k"
    )
)
MODEL_ROOT = Path(os.environ.get("PBMC3K_MODEL_ROOT", PROJECT_ROOT / "models"))
DATASET_SEED = 42
TRIPLET_SEED = 42
TRAINING_SEEDS_BY_LENGTH = {
    10: [73, 101, 202],
    20: [73, 101, 202],
    50: [73, 101, 202],
    100: [73, 101, 202],
    200: [73, 101, 202],
}
ANNOTATION_COLUMN = "louvain"
BASE_MODEL = "neuml/pubmedbert-base-embeddings"


@dataclass(frozen=True)
class StudyRun:
    cs_length: int
    training_seed: int

    @property
    def dataset_id(self) -> str:
        return f"pbmc3k_cs{self.cs_length}"

    @property
    def model_id(self) -> str:
        return f"{self.dataset_id}_seed{self.training_seed}"

    @property
    def data_dir(self) -> Path:
        return STUDY_ROOT / "datasets" / self.dataset_id

    @property
    def model_dir(self) -> Path:
        return MODEL_ROOT / self.model_id

    @property
    def triplet_train_dir(self) -> Path:
        return self.data_dir / "datasets" / "scrna_train_MNR_hnm"

    @property
    def triplet_eval_dir(self) -> Path:
        return self.data_dir / "datasets" / "scrna_eval_MNR_hnm"


def build_study_runs() -> list[StudyRun]:
    return [
        StudyRun(cs_length=cs_length, training_seed=training_seed)
        for cs_length, training_seeds in TRAINING_SEEDS_BY_LENGTH.items()
        for training_seed in training_seeds
    ]


def make_scrna_config(run: StudyRun) -> DatascRNAConfig:
    return DatascRNAConfig(
        dataset_id=run.dataset_id,
        annotation_column=ANNOTATION_COLUMN,
        split_strategy="random_stratified",
        test_size=0.20,
        total_cells=None,
        output_dir=run.data_dir,
        source="scanpy.datasets.pbmc3k_processed",
        save_artifacts=True,
        preprocessing=False,
        highly_variable_genes=True,
        housekeeping_genes=True,
        cs_length=[run.cs_length],
        semantic=True,
        random_state=DATASET_SEED,
    )


def make_triplet_config(run: StudyRun) -> TripletGenerationConfig:
    return TripletGenerationConfig(
        annotation_column=ANNOTATION_COLUMN,
        hard_negative_mining=True,
        loss="MNR",
        testrun=False,
        hf_upload=False,
        hf_name=None,
        model=BASE_MODEL,
        batch_size=256,
        subset_size=5000,
        seed=TRIPLET_SEED,
    )


def make_training_config(run: StudyRun) -> TrainingSTConfig:
    return TrainingSTConfig(
        model=BASE_MODEL,
        loss="MNR",
        save_to_local=True,
        save_to_hf=False,
        new_model_name=run.model_id,
        output_path=str(run.model_dir),
        batch_size=64,
        epochs=10,
        learning_rate=1e-5,
        logging_steps=100,
        save_strategy="no",
        save_epoch_models=True,
        load_from_hf=False,
        scrna_hf_dataset=None,
        ncbi_hf_dataset=None,
        seed=run.training_seed,
    )


def load_pbmc3k():
    adata = sc.datasets.pbmc3k_processed()
    adata.obs_names_make_unique()
    if ANNOTATION_COLUMN not in adata.obs:
        raise ValueError(f"PBMC3k AnnData is missing obs column: {ANNOTATION_COLUMN}")
    if "highly_variable" not in adata.var:
        adata.var["highly_variable"] = True
    return adata


def write_study_manifest(runs: list[StudyRun]) -> Path:
    STUDY_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "study_id": STUDY_ID,
        "source": "scanpy.datasets.pbmc3k_processed",
        "annotation_column": ANNOTATION_COLUMN,
        "dataset_seed": DATASET_SEED,
        "triplet_seed": TRIPLET_SEED,
        "base_model": BASE_MODEL,
        "runs": [
            {
                **asdict(run),
                "dataset_id": run.dataset_id,
                "model_id": run.model_id,
                "data_dir": str(run.data_dir),
                "model_dir": str(run.model_dir),
            }
            for run in runs
        ],
    }
    manifest_path = STUDY_ROOT / "study_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest_path


def generate_dataset_and_triplets(run: StudyRun, adata) -> None:
    if run.data_dir.exists():
        raise FileExistsError(
            f"Dataset already exists: {run.data_dir}. Set PBMC3K_STUDY_ROOT for a fresh run."
        )
    scrna_config = make_scrna_config(run)
    dataset_dict, _ = build_datasets(
        adata=adata,
        datasets=["scrna"],
        scrna_config=scrna_config,
        train_semantic=True,
        test_semantic=False,
    )

    triplet_config = make_triplet_config(run)
    train_datasets = build_triplets(
        dataset_dict=dataset_dict,
        triplet_config=triplet_config,
    )

    run.triplet_train_dir.parent.mkdir(parents=True, exist_ok=True)
    train_datasets["scrna"]["train_MNR_hnm"].save_to_disk(str(run.triplet_train_dir))
    train_datasets["scrna"]["eval_MNR_hnm"].save_to_disk(str(run.triplet_eval_dir))


def load_triplet_dataset(run: StudyRun) -> dict:
    return {
        "scrna": {
            "train_MNR_hnm": load_from_disk(str(run.triplet_train_dir)),
            "eval_MNR_hnm": load_from_disk(str(run.triplet_eval_dir)),
        }
    }


def train_run(run: StudyRun) -> None:
    if (run.model_dir / "model.safetensors").exists():
        raise FileExistsError(
            f"Model already exists: {run.model_dir}. Set PBMC3K_MODEL_ROOT for a fresh run."
        )
    train_model(
        dataset_dict=load_triplet_dataset(run),
        datasets=["scrna"],
        train_config=make_training_config(run),
        hf_upload=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PBMC3k cell sentence length and seed stability study."
    )
    parser.add_argument(
        "--stage",
        choices=("generate", "train", "all"),
        default="all",
        help="Run dataset/triplet generation, model training, or both.",
    )
    parser.add_argument(
        "--cs-length",
        type=int,
        choices=tuple(TRAINING_SEEDS_BY_LENGTH),
        help="Restrict execution to one cell sentence length.",
    )
    parser.add_argument(
        "--training-seed",
        type=int,
        help="Restrict execution to one training seed. Use with --cs-length for one model.",
    )
    return parser.parse_args()


def filter_runs(runs: list[StudyRun], *, cs_length: int | None, training_seed: int | None) -> list[StudyRun]:
    selected = runs
    if cs_length is not None:
        selected = [run for run in selected if run.cs_length == cs_length]
    if training_seed is not None:
        selected = [run for run in selected if run.training_seed == training_seed]
    if not selected:
        raise ValueError("No PBMC3k stability runs matched the requested filters.")
    return selected


def main() -> None:
    args = parse_args()
    runs = filter_runs(
        build_study_runs(),
        cs_length=args.cs_length,
        training_seed=args.training_seed,
    )
    write_study_manifest(runs)

    if args.stage in ("generate", "all"):
        adata = load_pbmc3k()
        generated_lengths = set()
        for run in runs:
            if run.cs_length in generated_lengths:
                continue
            generate_dataset_and_triplets(run, adata)
            generated_lengths.add(run.cs_length)
        del adata
        gc.collect()

    if args.stage in ("train", "all"):
        for run in runs:
            train_run(run)


if __name__ == "__main__":
    main()
