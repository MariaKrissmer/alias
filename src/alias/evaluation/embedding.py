from dataclasses import dataclass, asdict
from datasets import Dataset
from typing import Optional, List, Literal, Dict, Tuple, Any
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import re
import json

from alias.util.artifacts import (
    create_run_directory,
    load_annotation_map,
    load_embedding_frame,
    save_embedding_frame,
    write_metadata,
)


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


def resolve_artifact_root(output_dir: str | Path | None, category: str) -> Path:
    """Preserve existing category-specific roots while supporting generic output roots."""
    base_dir = Path(output_dir or "_out")
    if base_dir.name == category:
        return base_dir
    return base_dir / category


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
        source_index_column = "index" if embedding_config.index and "index" in df_cells.columns else None
        if source_index_column is not None:
            df_cells.index = df_cells[source_index_column].astype(str)
        else:
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


def load_saved_embeddings(
    embedding_path: str | Path,
    annotation_map_path: str | Path | None = None,
    annotation_column: str | None = None,
) -> pd.DataFrame | dict[str, Any]:
    """Load saved embedding metadata or one embedding frame."""
    embedding_path = Path(embedding_path)

    if embedding_path.is_dir():
        metadata_path = embedding_path / "metadata.json"
        if not metadata_path.exists():
            metadata_path = embedding_path / "embedding_metadata.json"

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        if isinstance(metadata, dict) and "artifacts" in metadata and "dataset_name" in metadata:
            return {metadata["dataset_name"]: metadata["artifacts"]}

        return metadata

    if embedding_path.suffix == ".json":
        with embedding_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        if isinstance(metadata, dict) and "artifacts" in metadata and "dataset_name" in metadata:
            return {metadata["dataset_name"]: metadata["artifacts"]}

        return metadata

    embedding_df = load_embedding_frame(embedding_path)
    embedding_df.index = embedding_df.index.astype(str)

    annotation_map = load_annotation_map(annotation_map_path)
    if annotation_map is None:
        return embedding_df

    if all(isinstance(value_map, dict) for value_map in annotation_map.values()):
        for column_name, value_map in annotation_map.items():
            embedding_df[column_name] = embedding_df.index.map(
                lambda idx: value_map.get(idx, "unknown")
            )
        return embedding_df

    inferred_column = annotation_column
    if inferred_column is None and annotation_map_path is not None:
        annotation_name = Path(annotation_map_path).name
        if annotation_name.endswith("_input_mapping.json"):
            inferred_column = "data"
        elif annotation_name.startswith("df_celltypes"):
            inferred_column = "cell_type"
        elif annotation_name.startswith("df_labels"):
            inferred_column = "label"
        else:
            inferred_column = "annotation"

    embedding_df[inferred_column or "annotation"] = embedding_df.index.map(
        lambda idx: annotation_map.get(idx, "unknown")
    )

    return embedding_df

def load_embedding_model(model_name: str):
    from alias.util.load_hf_model import load_model

    return load_model(model_name)

def generate_embeddings(
    evaluation_dict: Dict[str, Dict[str, Any]],
    embedding_config: GenEmbeddingsConfig,
    **kwargs
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Generate and save embeddings for each dataset and entity type.
    Returns metadata with paths to saved Parquet files.
    """
    run_timestamp = kwargs.pop("run_timestamp", kwargs.pop("timestamp", None))

    cfg = asdict(embedding_config)
    cfg.update(kwargs)
    embedding_config = GenEmbeddingsConfig(**cfg)

    dfs_dict = prepare_dfs(evaluation_dict, embedding_config)
    embeddings_dict: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for model_name in tqdm(embedding_config.embedding_models, desc="Embedding Models"):
        print(f"\n Generating embeddings with model: {model_name}")
        st_model = load_embedding_model(model_name)
        cleaned_name = clean_model_name(model_name)
        model_metadata = {}
        embeddings_root = resolve_artifact_root(embedding_config.output_dir, "embeddings")

        for dataset_name, dataset_dfs in dfs_dict.items():
            run_dir = create_run_directory(
                root_dir=embeddings_root,
                category="",
                dataset_name=dataset_name,
                model_name=cleaned_name,
                timestamp=run_timestamp,
            )
            dataset_metadata = {}

            for key, (column_name, df) in dataset_dfs.items():
                texts = df[column_name].astype(str).tolist()
                print(f"Encoding {key} ({len(texts)} samples)")
                
                emb = sentence_transformer_embeddings(texts, st_model, embedding_config)
                emb_array = np.array(emb)

                emb_df = pd.DataFrame(emb_array)
                emb_df.index = df.index
                emb_df.index.name = "cell_id"

                annotation_map = None
                annotation_file_name = None
                if embedding_config.annotation_column in df:
                    annotation_map = df[[embedding_config.annotation_column]].to_dict()
                    annotation_file_name = f"{key}_annotations.json"

                if "cell_type" in df:
                    annotation_map = df["cell_type"].to_dict()
                    annotation_file_name = f"{key}_annotations.json"

                if embedding_config.additional_data is not None and key == "df_additional":
                    annotation_map = df[column_name].astype(str).to_dict()
                    annotation_file_name = f"{key}_input_mapping.json"

                meta_info = save_embedding_frame(
                    run_dir,
                    key,
                    emb_df,
                    annotation_map=annotation_map,
                    annotation_file_name=annotation_file_name,
                )
                meta_info.update(
                    {
                        "dataset": dataset_name,
                        "entity_type": key,
                        "column": column_name,
                        "run_dir": str(run_dir),
                    }
                )

                dataset_metadata[key] = meta_info

            model_metadata[dataset_name] = dataset_metadata
            write_metadata(
                run_dir,
                {
                    "dataset_name": dataset_name,
                    "model_name": cleaned_name,
                    "annotation_column": embedding_config.annotation_column,
                    "run_timestamp": Path(run_dir).name,
                    "artifacts": dataset_metadata,
                },
            )
            with (run_dir / "embedding_metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(dataset_metadata, handle, indent=2)

        embeddings_dict[cleaned_name] = model_metadata

    return embeddings_dict
