from dataclasses import dataclass, asdict
from typing import Optional, Literal, Union
from pathlib import Path
import json
import platform
import subprocess
import sys
import pandas as pd
from sentence_transformers import losses
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.evaluation import TripletEvaluator
import torch
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from alias.util.load_hf_model import load_model, load_hf_dataset
from tqdm import tqdm
from datasets import Dataset
from alias.util.hf_config import hf_config

@dataclass
class TrainingSTConfig:
    model: str
    loss: Literal['MNR', 'Triplet', 'Contrastive']
    
    save_to_local: bool = True
    save_to_hf: bool = False
    matryoshka: Optional[list[int]] = None

    # training hyperparams
    new_model_name: Optional[str] = None
    file_path: Optional[str] = None
    batch_size: int = 64
    epochs: int = 5
    semantic: bool = False
    save_steps: int = 10000
    save_strategy: str = "epoch"
    save_total_limit: int = 10
    save_epoch_models: bool = True
    warmup_steps: int = 1000
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "linear"
    outer_learning_rate_schedule: Literal["constant", "linear", "exponential", "explicit"] = "constant"
    lr_decay_gamma: float = 1.0
    min_learning_rate: Optional[float] = None
    epoch_learning_rates: Optional[list[float]] = None
    warmup_first_epoch_only: bool = False
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 100
    fp16: bool = False
    seed: int = 73
    index: bool = False
    testrun: bool = False
    
    load_from_hf: bool = True
    scrna_hf_dataset: str = None
    ncbi_hf_dataset: str = None
    
    output_path: str = None
    save_metadata: bool = True
    metadata_dir_name: str = "metadata"
    
    def __post_init__(self):
        """Validate that at least one save option is enabled."""
        if not (self.save_to_local or self.save_to_hf):
            raise ValueError(
                "Invalid configuration: at least one of `save_to_local` or `save_to_hf` must be used."
            )


def setup_loss(train_config: TrainingSTConfig, model: str):
    
    if train_config.loss == 'MNR':
        
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
        
        print('MultipleNegativesRanking Loss loaded!')
        
    elif train_config.loss == 'Triplet':
        
        train_loss = losses.TripletLoss(model=model)
        
        print('Triplet Loss loaded!')
        
    elif train_config.loss == 'Contrastive':
        
        train_loss = losses.ContrastiveLoss(model=model)
        
        print('Contrastive Loss loaded!')
    
    if train_config.matryoshka is not None:
        
        train_loss = losses.MatryoshkaLoss(model, train_loss, train_config.matryoshka)
        
        print(f"Matryoshka Loss with dimensions {train_config.matryoshka} loaded!")
        
    return train_loss


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
    return path


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _git_metadata() -> dict:
    metadata = {"commit": None, "dirty": None}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return metadata

    metadata["commit"] = commit.stdout.strip()
    metadata["dirty"] = bool(status.stdout.strip())
    return metadata


def _find_split(dataset: dict, split_name_fragment: str):
    return next(
        ((name, split) for name, split in dataset.items() if split_name_fragment in name),
        (None, None),
    )


def _dataset_columns(dataset_split) -> list[str]:
    if dataset_split is None:
        return []
    return list(getattr(dataset_split, "column_names", []) or [])


def _dataset_len(dataset_split) -> int | None:
    if dataset_split is None:
        return None
    return len(dataset_split)


def _source_for_dataset(dataset_name: str, cfg: dict) -> str | None:
    return cfg.get(f"{dataset_name}_hf_dataset")


def _summarize_training_datasets(dataset_dict: dict, datasets: list[str], cfg: dict) -> dict:
    required_columns = ["sentence1", "sentence2", "negative", "label"]
    summary = {}

    for dataset_name in datasets:
        dataset = dataset_dict.get(dataset_name, {})
        train_split_name, train_dataset = _find_split(dataset, "train")
        eval_split_name, eval_dataset = _find_split(dataset, "eval")
        columns = sorted(set(_dataset_columns(train_dataset)) | set(_dataset_columns(eval_dataset)))

        summary[dataset_name] = {
            "source": _source_for_dataset(dataset_name, cfg),
            "train_split": train_split_name,
            "eval_split": eval_split_name,
            "n_train_rows": _dataset_len(train_dataset),
            "n_eval_rows": _dataset_len(eval_dataset),
            "columns": columns,
            "required_triplet_columns_present": [
                column for column in required_columns if column in columns
            ],
            "required_triplet_columns_missing": [
                column for column in required_columns if column not in columns
            ],
        }

    return summary


def _training_progress_frame(log_history: list[dict]) -> pd.DataFrame:
    if not log_history:
        return pd.DataFrame()
    return pd.DataFrame(log_history)


def write_training_metadata(
    *,
    output_path: Path | str,
    train_config: TrainingSTConfig,
    cfg: dict,
    dataset_dict: dict,
    datasets: list[str],
    output_model_name: str,
    device: str | torch.device,
    log_history: list[dict] | None = None,
    evaluation_summary: dict | None = None,
) -> dict[str, str]:
    if not cfg.get("save_metadata", True):
        return {}

    output_path = Path(output_path)
    metadata_dir = output_path / cfg.get("metadata_dir_name", "metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)

    effective_config = asdict(train_config)
    effective_config.update(cfg)

    training_metadata = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_path": str(output_path),
        "output_model_name": output_model_name,
        "hf_model_repo": output_model_name if cfg.get("save_to_hf") else None,
        "datasets": list(datasets),
        "device": str(device),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "sentence_transformers_version": _package_version("sentence-transformers"),
        "datasets_version": _package_version("datasets"),
        "git": _git_metadata(),
    }

    artifact_paths = {
        "training_config": _write_json(
            metadata_dir / "training_config.json",
            effective_config,
        ),
        "training_metadata": _write_json(
            metadata_dir / "training_metadata.json",
            training_metadata,
        ),
        "dataset_metadata": _write_json(
            metadata_dir / "dataset_metadata.json",
            _summarize_training_datasets(dataset_dict, datasets, cfg),
        ),
        "evaluation_summary": _write_json(
            metadata_dir / "evaluation_summary.json",
            evaluation_summary or {},
        ),
    }

    progress_path = metadata_dir / "training_progress.csv"
    _training_progress_frame(log_history or []).to_csv(progress_path, index=False)
    artifact_paths["training_progress"] = progress_path

    return {key: str(path) for key, path in artifact_paths.items()}


def _evaluator_metrics(evaluator, model) -> dict:
    result = evaluator(model)
    if isinstance(result, dict):
        return result
    return {"score": result}


def _train_output_metrics(train_output) -> dict:
    metrics = getattr(train_output, "metrics", None)
    return metrics or {}


def _best_checkpoint(trainer) -> str | None:
    return getattr(trainer.state, "best_model_checkpoint", None)


def _cfg_get(train_config: TrainingSTConfig | dict, key: str, default=None):
    if isinstance(train_config, dict):
        return train_config.get(key, default)
    return getattr(train_config, key, default)


def _learning_rate_for_outer_epoch(
    train_config: TrainingSTConfig | dict,
    *,
    epoch_index: int,
) -> float:
    base_lr = float(_cfg_get(train_config, "learning_rate"))
    schedule = _cfg_get(train_config, "outer_learning_rate_schedule", "constant")
    epoch_learning_rates = _cfg_get(train_config, "epoch_learning_rates", None)
    min_lr = _cfg_get(train_config, "min_learning_rate", None)

    if schedule == "explicit":
        if not epoch_learning_rates:
            raise ValueError(
                "`epoch_learning_rates` must be set when "
                "`outer_learning_rate_schedule='explicit'`."
            )
        if epoch_index >= len(epoch_learning_rates):
            raise ValueError(
                f"Missing explicit learning rate for epoch index {epoch_index}; "
                f"only {len(epoch_learning_rates)} rates were provided."
            )
        lr = float(epoch_learning_rates[epoch_index])
    elif schedule == "linear":
        epochs = int(_cfg_get(train_config, "epochs", 1))
        if epochs <= 1:
            lr = base_lr
        else:
            target_lr = float(min_lr) if min_lr is not None else 0.0
            progress = epoch_index / float(max(1, epochs - 1))
            lr = base_lr + (target_lr - base_lr) * progress
    elif schedule == "exponential":
        gamma = float(_cfg_get(train_config, "lr_decay_gamma", 1.0))
        lr = base_lr * (gamma**epoch_index)
    elif schedule == "constant":
        lr = base_lr
    else:
        raise ValueError(f"Unknown outer learning rate schedule: {schedule}")

    if min_lr is not None:
        lr = max(lr, float(min_lr))
    return lr


def _warmup_steps_for_outer_epoch(
    train_config: TrainingSTConfig | dict,
    *,
    epoch_index: int,
) -> int:
    warmup_steps = int(_cfg_get(train_config, "warmup_steps", 0))
    if _cfg_get(train_config, "warmup_first_epoch_only", False) and epoch_index > 0:
        return 0
    return warmup_steps


def _lr_scheduler_type_for_outer_epoch(
    train_config: TrainingSTConfig | dict,
    *,
    epoch_index: int,
) -> str:
    return _cfg_get(train_config, "lr_scheduler_type", "linear")


def setup_train(dataset_dict: dict, datasets: str, train_config: TrainingSTConfig, **kwargs):
    cfg = asdict(train_config)
    cfg.update(kwargs)

    dataset = dataset_dict[datasets]

    train_dataset = next((v for k, v in dataset.items() if "train" in k), None)
    eval_dataset = next((v for k, v in dataset.items() if "eval" in k), None)

    if train_dataset is None or eval_dataset is None:
        raise ValueError(f"Could not find train/eval datasets for {datasets}")

    model = load_model(cfg["model"])

    # move model to device
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    )
    model.to(device)
    print(f"Loaded model {cfg['model']} on {device}")

    train_loss = setup_loss(train_config, model)

    if cfg["output_path"] is None:
        date_str = datetime.now().strftime("%Y%m%d")
        output_model_name = cfg.get("new_model_name") or f"{cfg['model'].split('/')[-1]}_{datasets}_{date_str}"
        # Use current working directory instead of __file__ location
        # This works for both pip-installed and editable installs
        output_path = Path.cwd() / "models" / output_model_name

    else:
        output_path = Path(cfg["output_path"])
        output_model_name = cfg.get("new_model_name") or output_path.name
        
    output_path.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=cfg["epochs"],
        warmup_steps=cfg["warmup_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        per_device_train_batch_size=cfg["batch_size"],
        logging_dir="./logs",
        fp16=cfg["fp16"],
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy=cfg["save_strategy"],
        save_total_limit=cfg["save_total_limit"],
        max_grad_norm=cfg["max_grad_norm"],
        seed=cfg["seed"],
        data_seed=cfg["seed"],
    )

    # Evaluator
    triplet_evaluator = TripletEvaluator(
        anchors=eval_dataset[:1000]["sentence1"],
        positives=eval_dataset[:1000]["sentence2"],
        negatives=eval_dataset[:1000]["negative"],
        name=f"triplet_eval_{datasets}",
    )
    initial_metrics = _evaluator_metrics(triplet_evaluator, model)
    
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=triplet_evaluator,
    )

    print(f"Starting fine-tuning on {datasets}")
    train_output = trainer.train()
    print("Training completed.")

    evaluation_summary = {
        "evaluator": "TripletEvaluator",
        "eval_subset_size": min(1000, len(eval_dataset)),
        "initial_metrics": initial_metrics,
        "final_train_metrics": _train_output_metrics(train_output),
        "best_model_checkpoint": _best_checkpoint(trainer),
    }
    write_training_metadata(
        output_path=output_path,
        train_config=train_config,
        cfg=cfg,
        dataset_dict={datasets: dataset},
        datasets=[datasets],
        output_model_name=output_model_name,
        device=device,
        log_history=trainer.state.log_history,
        evaluation_summary=evaluation_summary,
    )

    if cfg["save_to_local"]:
        model.save_pretrained(str(output_path))
        print(f" Model saved to {output_path}")

    if cfg["save_to_hf"]:
        model.push_to_hub(repo_id=output_model_name, token=hf_config.HF_TOKEN_UPLOAD, private=True)
        print(f"Model pushed to: https://huggingface.co/{output_model_name}")
    
    return model
        
        
def setup_train_multi_dataset(dataset_dict: dict, datasets: list[str], train_config: TrainingSTConfig, **kwargs):
    cfg = asdict(train_config)
    cfg.update(kwargs)

    print(f"Training alternately per epoch on: {datasets}")

    model = load_model(cfg["model"])
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    )
    model.to(device)

    train_loss = setup_loss(train_config, model)
    
    if cfg["output_path"] is None:
        date_str = datetime.now().strftime("%Y%m%d")
        output_model_name = cfg.get("new_model_name") or f"{cfg['model'].split('/')[-1]}_multi_{date_str}"
        # Use current working directory instead of __file__ location
        # This works for both pip-installed and editable installs
        output_path = Path.cwd() / "models" / output_model_name

    else:
        output_path = Path(cfg["output_path"])
        output_model_name = cfg.get("new_model_name") or output_path.name
        
    output_path.mkdir(parents=True, exist_ok=True)

    metadata_dataset_dict = {}
    log_history = []
    evaluation_summaries = []

    for epoch in range(cfg["epochs"]):
        current_dataset_name = datasets[epoch % len(datasets)]
        current_learning_rate = _learning_rate_for_outer_epoch(cfg, epoch_index=epoch)
        current_warmup_steps = _warmup_steps_for_outer_epoch(cfg, epoch_index=epoch)
        current_lr_scheduler_type = _lr_scheduler_type_for_outer_epoch(cfg, epoch_index=epoch)
        
        print(
            f"Epoch {epoch+1}/{cfg['epochs']} — training on {current_dataset_name} "
            f"(learning_rate={current_learning_rate:.3g}, "
            f"warmup_steps={current_warmup_steps}, "
            f"lr_scheduler_type={current_lr_scheduler_type})"
        )
        
        if cfg["load_from_hf"]:
            if current_dataset_name == "scrna":
                ds = load_hf_dataset(cfg["scrna_hf_dataset"])
            elif current_dataset_name == "ncbi":
                ds = load_hf_dataset(cfg["ncbi_hf_dataset"])
            else:
                # Fallback to local dataset if dataset name not recognized
                ds = dataset_dict[current_dataset_name]
        else:
            ds = dataset_dict[current_dataset_name]
        metadata_dataset_dict[current_dataset_name] = ds

        train_dataset = next((v for k, v in ds.items() if "train" in k), None)
        eval_dataset = next((v for k, v in ds.items() if "eval" in k), None)
        if train_dataset is None or eval_dataset is None:
            raise ValueError(f"Missing train/eval split for {current_dataset_name}")
        

        subset = min(1000, len(eval_dataset))
        evaluator = TripletEvaluator(
            anchors=eval_dataset.select(range(subset))["sentence1"],
            positives=eval_dataset.select(range(subset))["sentence2"],
            negatives=eval_dataset.select(range(subset))["negative"],
            name=f"triplet_eval_{current_dataset_name}",
        )
        initial_metrics = _evaluator_metrics(evaluator, model)

        training_args = SentenceTransformerTrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=1,
            per_device_train_batch_size=cfg["batch_size"],
            warmup_steps=current_warmup_steps,
            learning_rate=current_learning_rate,
            lr_scheduler_type=current_lr_scheduler_type,
            logging_dir="./logs",
            fp16=cfg["fp16"],
            logging_steps=cfg["logging_steps"],
            eval_strategy="epoch",
            save_strategy=cfg["save_strategy"],
            save_total_limit=cfg["save_total_limit"],
            max_grad_norm=cfg["max_grad_norm"],
            seed=cfg["seed"],
            data_seed=cfg["seed"],
        )
        
        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=train_loss,
            evaluator=evaluator,
        )

        train_output = trainer.train(resume_from_checkpoint=False)
        epoch_log_history = []
        for log_entry in trainer.state.log_history:
            annotated_log_entry = dict(log_entry)
            annotated_log_entry["outer_epoch"] = epoch + 1
            annotated_log_entry["dataset"] = current_dataset_name
            annotated_log_entry["configured_learning_rate"] = current_learning_rate
            annotated_log_entry["configured_warmup_steps"] = current_warmup_steps
            annotated_log_entry["configured_lr_scheduler_type"] = current_lr_scheduler_type
            epoch_log_history.append(annotated_log_entry)
        log_history.extend(epoch_log_history)
        evaluation_summaries.append(
            {
                "epoch": epoch + 1,
                "dataset": current_dataset_name,
                "configured_learning_rate": current_learning_rate,
                "configured_warmup_steps": current_warmup_steps,
                "configured_lr_scheduler_type": current_lr_scheduler_type,
                "evaluator": "TripletEvaluator",
                "eval_subset_size": subset,
                "initial_metrics": initial_metrics,
                "final_train_metrics": _train_output_metrics(train_output),
                "best_model_checkpoint": _best_checkpoint(trainer),
            }
        )
        if cfg["save_epoch_models"]:
            model.save_pretrained(str(output_path / f"epoch_{epoch+1}_{current_dataset_name}"))

    write_training_metadata(
        output_path=output_path,
        train_config=train_config,
        cfg=cfg,
        dataset_dict=metadata_dataset_dict,
        datasets=datasets,
        output_model_name=output_model_name,
        device=device,
        log_history=log_history,
        evaluation_summary={
            "mode": "multi_dataset",
            "epochs": evaluation_summaries,
        },
    )

    if cfg["save_to_local"]:
        model.save_pretrained(str(output_path))
        print(f"Final model saved to {output_path}")

    if cfg["save_to_hf"]:
        # Lazy import to avoid requiring tokens when not uploading
        from alias.util.hf_config import hf_config
        model.push_to_hub(repo_id=output_model_name, token=hf_config.HF_TOKEN_UPLOAD, private=True)
        print(f"Model pushed to: https://huggingface.co/{output_model_name}")
    
    return model


def train_model(
    dataset_dict: dict, 
    datasets: Union[str, list[str]], 
    train_config: TrainingSTConfig, 
    **kwargs
):
    """
    Train a model on one or more datasets.
    
    Args:
        dataset_dict: Dictionary containing training datasets
        datasets: Single dataset name ('scrna') or list of names (['scrna', 'ncbi'])
        train_config: Training configuration
        **kwargs: Additional arguments to override config
    
    Returns:
        Trained model
    """
    # Normalize datasets to list
    if isinstance(datasets, str):
        datasets = [datasets]
    
    if len(datasets) == 1:
        return setup_train(dataset_dict, datasets[0], train_config, **kwargs)
    else:
        return setup_train_multi_dataset(dataset_dict, datasets, train_config, **kwargs)
