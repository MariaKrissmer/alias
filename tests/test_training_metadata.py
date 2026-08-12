import json
from types import SimpleNamespace
from pathlib import Path

import pandas as pd
from datasets import Dataset


def _triplet_dataset(n_rows: int = 4) -> Dataset:
    return Dataset.from_dict(
        {
            "sentence1": [f"anchor {i}" for i in range(n_rows)],
            "sentence2": [f"positive {i}" for i in range(n_rows)],
            "negative": [f"negative {i}" for i in range(n_rows)],
            "label": ["A", "B"] * (n_rows // 2),
        }
    )


def test_write_training_metadata_files_records_config_dataset_and_progress(tmp_path: Path):
    from alias.model.training import TrainingSTConfig, write_training_metadata

    train_config = TrainingSTConfig(
        model="neuml/pubmedbert-base-embeddings",
        loss="MNR",
        new_model_name="MG_HIAI5k_S2",
        save_to_local=True,
        save_to_hf=True,
        batch_size=64,
        epochs=1,
        load_from_hf=False,
        scrna_hf_dataset="mariakrissmer/scrna_HIAI5k_S2_heldout_donor_semantic",
    )
    dataset_dict = {
        "scrna": {
            "scrna_train": _triplet_dataset(6),
            "scrna_eval": _triplet_dataset(2),
        }
    }
    log_history = [
        {"loss": 0.8, "learning_rate": 1e-5, "epoch": 0.5, "step": 10},
        {"eval_loss": 0.7, "epoch": 1.0, "step": 20},
    ]

    artifact_paths = write_training_metadata(
        output_path=tmp_path,
        train_config=train_config,
        cfg={**train_config.__dict__},
        dataset_dict=dataset_dict,
        datasets=["scrna"],
        output_model_name="MG_HIAI5k_S2",
        device="mps",
        log_history=log_history,
        evaluation_summary={
            "evaluator": "TripletEvaluator",
            "eval_subset_size": 2,
            "metrics": {"triplet_eval_scrna_cosine_accuracy": 0.75},
        },
    )

    metadata_dir = tmp_path / "metadata"
    expected_files = {
        "training_config",
        "training_metadata",
        "dataset_metadata",
        "training_progress",
        "evaluation_summary",
    }
    assert set(artifact_paths) == expected_files
    for path in artifact_paths.values():
        assert Path(path).exists()
        assert Path(path).is_relative_to(metadata_dir)

    with (metadata_dir / "training_config.json").open("r", encoding="utf-8") as handle:
        config_payload = json.load(handle)
    assert config_payload["model"] == "neuml/pubmedbert-base-embeddings"
    assert config_payload["loss"] == "MNR"
    assert config_payload["save_metadata"] is True
    assert "HF_TOKEN_UPLOAD" not in json.dumps(config_payload)
    assert "HF_TOKEN_DOWNLOAD" not in json.dumps(config_payload)

    with (metadata_dir / "dataset_metadata.json").open("r", encoding="utf-8") as handle:
        dataset_payload = json.load(handle)
    scrna_payload = dataset_payload["scrna"]
    assert scrna_payload["source"] == "mariakrissmer/scrna_HIAI5k_S2_heldout_donor_semantic"
    assert scrna_payload["train_split"] == "scrna_train"
    assert scrna_payload["eval_split"] == "scrna_eval"
    assert scrna_payload["n_train_rows"] == 6
    assert scrna_payload["n_eval_rows"] == 2
    assert scrna_payload["required_triplet_columns_missing"] == []

    with (metadata_dir / "training_metadata.json").open("r", encoding="utf-8") as handle:
        training_payload = json.load(handle)
    assert training_payload["output_model_name"] == "MG_HIAI5k_S2"
    assert training_payload["datasets"] == ["scrna"]
    assert training_payload["device"] == "mps"

    progress = pd.read_csv(metadata_dir / "training_progress.csv")
    assert list(progress["step"]) == [10, 20]

    with (metadata_dir / "evaluation_summary.json").open("r", encoding="utf-8") as handle:
        evaluation_payload = json.load(handle)
    assert evaluation_payload["evaluator"] == "TripletEvaluator"
    assert evaluation_payload["metrics"]["triplet_eval_scrna_cosine_accuracy"] == 0.75


def test_write_training_metadata_reports_missing_triplet_columns(tmp_path: Path):
    from alias.model.training import TrainingSTConfig, write_training_metadata

    train_config = TrainingSTConfig(
        model="sentence-transformers/all-MiniLM-L6-v2",
        loss="MNR",
        output_path=str(tmp_path),
        save_to_hf=False,
    )
    incomplete = Dataset.from_dict({"sentence1": ["anchor"], "label": ["A"]})

    write_training_metadata(
        output_path=tmp_path,
        train_config=train_config,
        cfg={**train_config.__dict__},
        dataset_dict={"scrna": {"scrna_train": incomplete, "scrna_eval": incomplete}},
        datasets=["scrna"],
        output_model_name="test_model",
        device="cpu",
        log_history=[],
        evaluation_summary={},
    )

    with (tmp_path / "metadata" / "dataset_metadata.json").open("r", encoding="utf-8") as handle:
        dataset_payload = json.load(handle)

    assert dataset_payload["scrna"]["required_triplet_columns_present"] == ["sentence1", "label"]
    assert dataset_payload["scrna"]["required_triplet_columns_missing"] == ["sentence2", "negative"]


def test_write_training_metadata_can_be_disabled(tmp_path: Path):
    from alias.model.training import TrainingSTConfig, write_training_metadata

    train_config = TrainingSTConfig(
        model="sentence-transformers/all-MiniLM-L6-v2",
        loss="MNR",
        output_path=str(tmp_path),
        save_to_hf=False,
        save_metadata=False,
    )

    artifact_paths = write_training_metadata(
        output_path=tmp_path,
        train_config=train_config,
        cfg={**train_config.__dict__},
        dataset_dict={"scrna": {"scrna_train": _triplet_dataset(), "scrna_eval": _triplet_dataset()}},
        datasets=["scrna"],
        output_model_name="test_model",
        device="cpu",
        log_history=[],
        evaluation_summary={},
    )

    assert artifact_paths == {}
    assert not (tmp_path / "metadata").exists()


def test_train_model_single_dataset_writes_metadata_with_mocked_trainer(tmp_path: Path, monkeypatch):
    import alias.model.training as training
    from alias.model.training import TrainingSTConfig, train_model

    class FakeModel:
        def to(self, device):
            self.device = device

        def save_pretrained(self, output_path):
            self.output_path = output_path

    class FakeEvaluator:
        def __init__(self, anchors, positives, negatives, name):
            self.anchors = anchors
            self.positives = positives
            self.negatives = negatives
            self.name = name

        def __call__(self, model):
            return {f"{self.name}_cosine_accuracy": 0.9}

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.state = SimpleNamespace(
                log_history=[{"loss": 0.5, "step": 1, "epoch": 1.0}],
                best_model_checkpoint="checkpoint-1",
            )

        def train(self, *args, **kwargs):
            return SimpleNamespace(metrics={"train_loss": 0.5})

    monkeypatch.setattr(training, "load_model", lambda model_name: FakeModel())
    monkeypatch.setattr(training, "setup_loss", lambda train_config, model: object())
    monkeypatch.setattr(training, "TripletEvaluator", FakeEvaluator)
    monkeypatch.setattr(training, "SentenceTransformerTrainer", FakeTrainer)

    dataset_dict = {
        "scrna": {
            "scrna_train": _triplet_dataset(4),
            "scrna_eval": _triplet_dataset(2),
        }
    }
    train_config = TrainingSTConfig(
        model="sentence-transformers/all-MiniLM-L6-v2",
        loss="MNR",
        new_model_name="mock_model",
        output_path=str(tmp_path),
        save_to_local=True,
        save_to_hf=False,
        load_from_hf=False,
        scrna_hf_dataset="mariakrissmer/scrna_mock",
    )

    model = train_model(
        dataset_dict=dataset_dict,
        datasets=["scrna"],
        train_config=train_config,
    )

    assert isinstance(model, FakeModel)
    metadata_dir = tmp_path / "metadata"
    assert (metadata_dir / "training_config.json").exists()
    assert (metadata_dir / "dataset_metadata.json").exists()
    assert (metadata_dir / "training_metadata.json").exists()
    assert (metadata_dir / "training_progress.csv").exists()
    assert (metadata_dir / "evaluation_summary.json").exists()

    with (metadata_dir / "evaluation_summary.json").open("r", encoding="utf-8") as handle:
        evaluation_summary = json.load(handle)
    assert evaluation_summary["initial_metrics"]["triplet_eval_scrna_cosine_accuracy"] == 0.9
    assert evaluation_summary["final_train_metrics"]["train_loss"] == 0.5
    assert evaluation_summary["best_model_checkpoint"] == "checkpoint-1"


def test_train_model_single_dataset_passes_learning_rate_and_max_grad_norm(
    tmp_path: Path, monkeypatch
):
    import alias.model.training as training
    from alias.model.training import TrainingSTConfig, train_model

    captured_args = {}

    class FakeModel:
        def to(self, device):
            self.device = device

        def save_pretrained(self, output_path):
            self.output_path = output_path

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            captured_args.update(kwargs)

    class FakeEvaluator:
        def __init__(self, anchors, positives, negatives, name):
            self.name = name

        def __call__(self, model):
            return {f"{self.name}_cosine_accuracy": 0.9}

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.state = SimpleNamespace(log_history=[], best_model_checkpoint=None)

        def train(self, *args, **kwargs):
            return SimpleNamespace(metrics={})

    monkeypatch.setattr(training, "load_model", lambda model_name: FakeModel())
    monkeypatch.setattr(training, "setup_loss", lambda train_config, model: object())
    monkeypatch.setattr(training, "TripletEvaluator", FakeEvaluator)
    monkeypatch.setattr(training, "SentenceTransformerTrainer", FakeTrainer)
    monkeypatch.setattr(training, "SentenceTransformerTrainingArguments", FakeTrainingArguments)

    dataset_dict = {
        "scrna": {
            "scrna_train": _triplet_dataset(4),
            "scrna_eval": _triplet_dataset(2),
        }
    }
    train_config = TrainingSTConfig(
        model="sentence-transformers/all-MiniLM-L6-v2",
        loss="MNR",
        output_path=str(tmp_path),
        save_to_hf=False,
        save_metadata=False,
        learning_rate=2e-5,
        max_grad_norm=0.5,
        seed=101,
    )

    train_model(dataset_dict=dataset_dict, datasets=["scrna"], train_config=train_config)

    assert captured_args["learning_rate"] == 2e-5
    assert captured_args["max_grad_norm"] == 0.5
    assert captured_args["seed"] == 101
    assert captured_args["data_seed"] == 101


def test_train_model_multi_dataset_passes_learning_rate_and_max_grad_norm(
    tmp_path: Path, monkeypatch
):
    import alias.model.training as training
    from alias.model.training import TrainingSTConfig, train_model

    captured_args = {}

    class FakeModel:
        def to(self, device):
            self.device = device

        def save_pretrained(self, output_path):
            self.output_path = output_path

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            captured_args.update(kwargs)

    class FakeEvaluator:
        def __init__(self, anchors, positives, negatives, name):
            self.name = name

        def __call__(self, model):
            return {f"{self.name}_cosine_accuracy": 0.9}

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.state = SimpleNamespace(log_history=[], best_model_checkpoint=None)

        def train(self, *args, **kwargs):
            return SimpleNamespace(metrics={})

    monkeypatch.setattr(training, "load_model", lambda model_name: FakeModel())
    monkeypatch.setattr(training, "setup_loss", lambda train_config, model: object())
    monkeypatch.setattr(training, "TripletEvaluator", FakeEvaluator)
    monkeypatch.setattr(training, "SentenceTransformerTrainer", FakeTrainer)
    monkeypatch.setattr(training, "SentenceTransformerTrainingArguments", FakeTrainingArguments)

    triplets = _triplet_dataset(4)
    dataset_dict = {
        "scrna": {"scrna_train": triplets, "scrna_eval": triplets.select(range(2))},
        "ncbi": {"ncbi_train": triplets, "ncbi_eval": triplets.select(range(2))},
    }
    train_config = TrainingSTConfig(
        model="sentence-transformers/all-MiniLM-L6-v2",
        loss="MNR",
        output_path=str(tmp_path),
        save_to_hf=False,
        save_metadata=False,
        load_from_hf=False,
        epochs=2,
        learning_rate=1e-5,
        max_grad_norm=0.75,
        seed=202,
    )

    train_model(
        dataset_dict=dataset_dict,
        datasets=["scrna", "ncbi"],
        train_config=train_config,
    )

    assert captured_args["learning_rate"] == 1e-5
    assert captured_args["max_grad_norm"] == 0.75
    assert captured_args["seed"] == 202
    assert captured_args["data_seed"] == 202


def test_train_model_multi_dataset_uses_new_model_name_by_default(
    tmp_path: Path, monkeypatch
):
    import alias.model.training as training
    from alias.model.training import TrainingSTConfig, train_model

    monkeypatch.chdir(tmp_path)

    class FakeModel:
        def to(self, device):
            self.device = device

        def save_pretrained(self, output_path):
            self.output_path = output_path

    class FakeEvaluator:
        def __init__(self, anchors, positives, negatives, name):
            self.name = name

        def __call__(self, model):
            return {f"{self.name}_cosine_accuracy": 0.9}

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.state = SimpleNamespace(log_history=[], best_model_checkpoint=None)

        def train(self, *args, **kwargs):
            return SimpleNamespace(metrics={})

    monkeypatch.setattr(training, "load_model", lambda model_name: FakeModel())
    monkeypatch.setattr(training, "setup_loss", lambda train_config, model: object())
    monkeypatch.setattr(training, "TripletEvaluator", FakeEvaluator)
    monkeypatch.setattr(training, "SentenceTransformerTrainer", FakeTrainer)

    triplets = _triplet_dataset(4)
    dataset_dict = {
        "ncbi": {"ncbi_train": triplets, "ncbi_eval": triplets.select(range(2))},
        "scrna": {"scrna_train": triplets, "scrna_eval": triplets.select(range(2))},
    }
    train_config = TrainingSTConfig(
        model="sentence-transformers/all-MiniLM-L6-v2",
        loss="MNR",
        new_model_name="MJ_HIAI5k_S2_N3",
        save_to_hf=False,
        load_from_hf=False,
        epochs=2,
    )

    train_model(
        dataset_dict=dataset_dict,
        datasets=["ncbi", "scrna"],
        train_config=train_config,
    )

    metadata_path = tmp_path / "models" / "MJ_HIAI5k_S2_N3" / "metadata" / "training_metadata.json"
    assert metadata_path.exists()
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["output_model_name"] == "MJ_HIAI5k_S2_N3"


def test_train_model_multi_dataset_can_disable_trainer_checkpoints_but_keep_epoch_saves(
    tmp_path: Path, monkeypatch
):
    import alias.model.training as training
    from alias.model.training import TrainingSTConfig, train_model

    captured_args = {}
    saved_paths = []

    class FakeModel:
        def to(self, device):
            self.device = device

        def save_pretrained(self, output_path):
            saved_paths.append(str(output_path))

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            captured_args.update(kwargs)

    class FakeEvaluator:
        def __init__(self, anchors, positives, negatives, name):
            self.name = name

        def __call__(self, model):
            return {f"{self.name}_cosine_accuracy": 0.9}

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.state = SimpleNamespace(log_history=[], best_model_checkpoint=None)

        def train(self, *args, **kwargs):
            return SimpleNamespace(metrics={})

    monkeypatch.setattr(training, "load_model", lambda model_name: FakeModel())
    monkeypatch.setattr(training, "setup_loss", lambda train_config, model: object())
    monkeypatch.setattr(training, "TripletEvaluator", FakeEvaluator)
    monkeypatch.setattr(training, "SentenceTransformerTrainer", FakeTrainer)
    monkeypatch.setattr(training, "SentenceTransformerTrainingArguments", FakeTrainingArguments)

    triplets = _triplet_dataset(4)
    dataset_dict = {
        "ncbi": {"ncbi_train": triplets, "ncbi_eval": triplets.select(range(2))},
        "scrna": {"scrna_train": triplets, "scrna_eval": triplets.select(range(2))},
    }
    train_config = TrainingSTConfig(
        model="sentence-transformers/all-MiniLM-L6-v2",
        loss="MNR",
        output_path=str(tmp_path),
        save_to_hf=False,
        save_metadata=False,
        load_from_hf=False,
        epochs=2,
        save_strategy="no",
        save_epoch_models=True,
    )

    train_model(
        dataset_dict=dataset_dict,
        datasets=["ncbi", "scrna"],
        train_config=train_config,
    )

    assert captured_args["save_strategy"] == "no"
    assert any(path.endswith("epoch_1_ncbi") for path in saved_paths)
    assert any(path.endswith("epoch_2_scrna") for path in saved_paths)
    assert str(tmp_path) in saved_paths
