from dataclasses import dataclass, asdict
from typing import Optional, Literal, Union
from pathlib import Path
import pandas as pd
from sentence_transformers import losses
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.evaluation import TripletEvaluator
import torch
from datetime import datetime
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
    save_total_limit: int = 10
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    logging_steps: int = 100
    fp16: bool = False
    seed: int = 73
    index: bool = False
    testrun: bool = False
    
    load_from_hf: bool = True
    scrna_hf_dataset: str = None
    ncbi_hf_dataset: str = None
    
    output_path: str = None
    
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
        output_dir = cfg.get("new_model_name") or f"{cfg['model'].split('/')[-1]}_{datasets}_{date_str}"
        # Use current working directory instead of __file__ location
        # This works for both pip-installed and editable installs
        output_path = Path.cwd() / "models" / output_dir

    else:
        output_path = Path(cfg["output_path"])
        
    output_path.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=cfg["epochs"],
        warmup_steps=cfg["warmup_steps"],
        per_device_train_batch_size=cfg["batch_size"],
        logging_dir="./logs",
        fp16=cfg["fp16"],
        logging_steps=cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="epoch",
        save_total_limit=cfg["save_total_limit"],
        max_grad_norm=1.0,
    )

    # Evaluator
    triplet_evaluator = TripletEvaluator(
        anchors=eval_dataset[:1000]["sentence1"],
        positives=eval_dataset[:1000]["sentence2"],
        negatives=eval_dataset[:1000]["negative"],
        name=f"triplet_eval_{datasets}",
    )
    triplet_evaluator(model)
    
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=triplet_evaluator,
    )

    print(f"Starting fine-tuning on {datasets}")
    trainer.train()
    print("Training completed.")

    if cfg["save_to_local"]:
        model.save_pretrained(str(output_path))
        print(f" Model saved to {output_path}")

    if cfg["save_to_hf"]:
        model.push_to_hub(repo_id=output_dir, token=hf_config.HF_TOKEN_UPLOAD, private=True)
        print(f"Model pushed to: https://huggingface.co/{output_dir}")
    
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
        output_dir = f"{cfg['model'].split('/')[-1]}_multi_{date_str}" or f"{cfg['new_model_name']}_multi_{date_str}"
        # Use current working directory instead of __file__ location
        # This works for both pip-installed and editable installs
        output_path = Path.cwd() / "models" / output_dir

    else:
        output_path = Path(cfg["output_path"])
        
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=1,
        per_device_train_batch_size=cfg["batch_size"],
        warmup_steps=cfg["warmup_steps"],
        logging_dir="./logs",
        fp16=cfg["fp16"],
        logging_steps=cfg["logging_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=cfg["save_total_limit"],
    )

    for epoch in range(cfg["epochs"]):
        current_dataset_name = datasets[epoch % len(datasets)]
        
        print(f"Epoch {epoch+1}/{cfg['epochs']} — training on {current_dataset_name}")
        
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
        
        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=train_loss,
            evaluator=evaluator,
        )

        trainer.train(resume_from_checkpoint=False)
        model.save_pretrained(str(output_path / f"epoch_{epoch+1}_{current_dataset_name}"))

    if cfg["save_to_local"]:
        model.save_pretrained(str(output_path))
        print(f"Final model saved to {output_path}")

    if cfg["save_to_hf"]:
        # Lazy import to avoid requiring tokens when not uploading
        from alias.util.hf_config import hf_config
        model.push_to_hub(repo_id=output_dir, token=hf_config.HF_TOKEN_UPLOAD, private=True)
        print(f"Model pushed to: https://huggingface.co/{output_dir}")
    
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