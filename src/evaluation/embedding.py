from dataclasses import dataclass, asdict
from datasets import Dataset
from typing import Optional, List, Literal, Dict, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import re
import json

from util.load_hf_model import load_model


# ---------------- CONFIG ---------------- #

@dataclass
class GenEmbeddingsConfig:
    annotation_column: str
    embedding_models: List[str]
    model_type: Literal['sentence_transformer']
    batch_size: Optional[int] = 64
    additional_data: Optional[List[str]] = None
    output_dir: Optional[str] = None
    max_cells: int = 20000
    index: bool = True

def clean_model_name(model_name: str) -> str:
    """Clean the model name by removing special characters except alphanumerics and underscores."""
    last_part = Path(model_name).name
    return re.sub(r"[^\w\d_]", "", last_part)


def sentence_transformer_embeddings(texts, st_model, embedding_config):
    """Generate embeddings for a batch of texts using a SentenceTransformer."""
    try:
        return st_model.encode(texts, batch_size=embedding_config.batch_size, show_progress_bar=True)
    except Exception as e:
        print(f"⚠️ Error generating batch embeddings: {e}")
        return np.zeros((len(texts), st_model.get_sentence_embedding_dimension()))

def prepare_dfs(
    evaluation_dict: Dict[str, Dict[str, Any]],
    embedding_config: GenEmbeddingsConfig
) -> Dict[str, Dict[str, Tuple[str, pd.DataFrame]]]:
    """
    Build nested dict of DataFrames for each dataset.
    Keeps index for cells.
    """
    dfs_dict = {}

    for dataset_name, split_dict in evaluation_dict.items():
        if "test" not in split_dict:
            print(f"Skipping {dataset_name}: no 'test' dataset found.")
            continue

        ds = split_dict["test"]
        df = pd.DataFrame(ds)

        # --- cells ---
        df_cells = df.copy()
        df_cells.index = df_cells.index.astype(str)
        if embedding_config.max_cells and len(df_cells) > embedding_config.max_cells:
            df_cells = df_cells.sample(n=embedding_config.max_cells, random_state=42)

        # --- genes ---
        genes = set()
        if "sentence1" in df:
            for sentence in df["sentence1"].dropna():
                genes.update(sentence.split())
        df_genes = pd.DataFrame({"gene": list(genes)})

        # --- cell types ---
        cell_types = df[embedding_config.annotation_column].unique() \
            if embedding_config.annotation_column in df else []
        df_celltypes = pd.DataFrame({"cell_type": cell_types})

        # --- labels ---
        labels = df["label"].unique() if "label" in df else []
        df_labels = pd.DataFrame({"label": labels})

        # --- structure ---
        dfs_dict[dataset_name] = {
            "df_cells": ("sentence1", df_cells),
            "df_genes": ("gene", df_genes),
            "df_celltypes": ("cell_type", df_celltypes),
            "df_labels": ("label", df_labels),
        }

        if embedding_config.additional_data:
            df_additional = pd.DataFrame(embedding_config.additional_data, columns=["data"])
            dfs_dict[dataset_name]["df_additional"] = ("data", df_additional)

    return dfs_dict

def generate_embeddings(
    evaluation_dict: Dict[str, Dict[str, Any]],
    embedding_config: GenEmbeddingsConfig,
    **kwargs
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Generate and save embeddings for each dataset and entity type.
    Returns metadata with paths to saved Parquet files.
    """
    cfg = asdict(embedding_config)
    cfg.update(kwargs)
    embedding_config = GenEmbeddingsConfig(**cfg)

    dfs_dict = prepare_dfs(evaluation_dict, embedding_config)
    embeddings_dict: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for model_name in tqdm(embedding_config.embedding_models, desc="Embedding Models"):
        print(f"\n Generating embeddings with model: {model_name}")
        st_model = load_model(model_name)
        cleaned_name = clean_model_name(model_name)

        model_dir = Path(embedding_config.output_dir or ".") / cleaned_name
        model_dir.mkdir(parents=True, exist_ok=True)

        model_metadata = {}

        for dataset_name, dataset_dfs in dfs_dict.items():
            dataset_dir = model_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            dataset_metadata = {}

            for key, (column_name, df) in dataset_dfs.items():
                texts = df[column_name].astype(str).tolist()
                print(f"Encoding {key} ({len(texts)} samples)")
                
                emb = sentence_transformer_embeddings(texts, st_model, embedding_config)
                emb_array = np.array(emb)

                # --- Save embeddings to Parquet ---
                emb_df = pd.DataFrame(emb_array)
                emb_df.index = df.index
                emb_df.index.name = "cell_id"
                out_path = dataset_dir / f"{key}.parquet"
                emb_df.to_parquet(out_path)

                # Save annotation map for the main annotation column
                ann_path = None
                if embedding_config.annotation_column in df:
                    annotation_map = df[[embedding_config.annotation_column]].to_dict()
                    ann_path = dataset_dir / f"{key}_annotations.json"
                    with open(ann_path, "w") as f:
                        json.dump(annotation_map, f, indent=2)

                # Save annotation map for cell type labels
                if 'cell_type' in df:
                    annotation_map = df['cell_type'].to_dict()
                    ann_path = dataset_dir / f"{key}_annotations.json"
                    with open(ann_path, "w") as f:
                        json.dump(annotation_map, f, indent=2)
                        
                if embedding_config.additional_data is not None and key == "df_additional":
                    mapping_dict = df[column_name].astype(str).to_dict()
                    ann_path = dataset_dir / f"{key}_input_mapping.json"
                    with open(ann_path, "w") as f:
                        json.dump(mapping_dict, f, indent=2)


                # --- Metadata only ---
                meta_info = {
                    "path": str(out_path),
                    "annotation_map": str(ann_path) if ann_path is not None else None,
                    "dataset": dataset_name,
                    "entity_type": key,
                    "column": column_name,
                    "n_samples": len(df),
                    "embedding_dim": emb_array.shape[1] if emb_array.size > 0 else 0,
                }

                dataset_metadata[key] = meta_info


            model_metadata[dataset_name] = dataset_metadata

        # Save JSON metadata for reproducibility
        with open(model_dir / "embedding_metadata.json", "w") as f:
            json.dump(model_metadata, f, indent=2)

        embeddings_dict[cleaned_name] = model_metadata

    return embeddings_dict
