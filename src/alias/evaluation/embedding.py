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
    force_regenerate: bool = False
    model_output_names: Optional[Dict[str, str]] = None

def clean_model_name(model_name: str) -> str:
    """Clean the model name by removing special characters except alphanumerics and underscores."""
    last_part = Path(model_name).name
    return re.sub(r"[^\w\d_]", "", last_part)


def output_model_name(model_name: str, embedding_config: GenEmbeddingsConfig) -> str:
    """Return the saved model key/path component for an embedding model."""
    if embedding_config.model_output_names and model_name in embedding_config.model_output_names:
        return clean_model_name(embedding_config.model_output_names[model_name])
    return clean_model_name(model_name)


def resolve_artifact_root(output_dir: str | Path | None, category: str) -> Path:
    """Preserve existing category-specific roots while supporting generic output roots."""
    base_dir = Path(output_dir or "_out")
    if base_dir.name == category:
        return base_dir
    return base_dir / category


def build_embedding_reuse_config(
    dataset_name: str,
    model_name: str,
    embedding_config: GenEmbeddingsConfig,
) -> dict[str, Any]:
    """Return the explicit config fields used to decide embedding reuse."""
    return {
        "dataset_name": dataset_name,
        "model_name": output_model_name(model_name, embedding_config),
        "annotation_column": embedding_config.annotation_column,
        "model_type": embedding_config.model_type,
        "batch_size": embedding_config.batch_size,
        "max_cells": embedding_config.max_cells,
        "index": embedding_config.index,
        "additional_data": list(embedding_config.additional_data or []),
    }


def build_embedding_reuse_signature(config_fields: dict[str, Any]) -> str:
    """Build a stable string signature for reuse matching."""
    return json.dumps(config_fields, sort_keys=True, separators=(",", ":"))


def find_reusable_embedding_run(
    embeddings_root: Path,
    dataset_name: str,
    model_name: str,
    embedding_config: GenEmbeddingsConfig,
) -> Path | None:
    """Return a previous run directory with matching explicit config, if any."""
    model_dir = embeddings_root / dataset_name / output_model_name(model_name, embedding_config)
    if not model_dir.exists():
        return None

    expected_fields = build_embedding_reuse_config(dataset_name, model_name, embedding_config)
    expected_signature = build_embedding_reuse_signature(expected_fields)

    candidate_dirs = sorted((path for path in model_dir.iterdir() if path.is_dir()), reverse=True)
    for candidate in candidate_dirs:
        metadata_path = candidate / "metadata.json"
        if not metadata_path.exists():
            continue

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        stored_signature = metadata.get("reuse_signature")
        stored_fields = metadata.get("reuse_config")
        if stored_signature == expected_signature:
            return candidate
        if stored_signature is None and stored_fields == expected_fields:
            return candidate

    return None


def sentence_transformer_embeddings(texts, st_model, embedding_config, *, batch_size: int | None = None):
    """Generate embeddings for a batch of texts using a SentenceTransformer."""
    effective_batch_size = batch_size if batch_size is not None else embedding_config.batch_size
    try:
        return st_model.encode(texts, batch_size=effective_batch_size, show_progress_bar=True)
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


def infer_embedding_run_timestamp(dataset_meta: dict[str, dict[str, Any]]) -> str | None:
    """Infer the embedding run timestamp from saved artifact metadata."""
    cell_meta = dataset_meta.get("df_cells", {})
    if "run_dir" in cell_meta:
        return Path(cell_meta["run_dir"]).name

    cell_umap = cell_meta.get("umap", {})
    if "path" in cell_umap:
        return Path(cell_umap["path"]).parent.name
    if "path" in cell_meta:
        return Path(cell_meta["path"]).parent.name
    return None


def load_embedding_artifact(
    artifact_meta: dict[str, Any],
    annotation_column: str | None = None,
) -> dict[str, Any]:
    """Load one saved embedding artifact with optional annotations and UMAP."""
    loaded_df = load_saved_embeddings(
        artifact_meta["path"],
        artifact_meta.get("annotation_map"),
        annotation_column=annotation_column,
    )
    if not isinstance(loaded_df, pd.DataFrame):
        raise TypeError("Expected a parquet-backed embedding artifact.")

    umap_df = None
    umap_meta = artifact_meta.get("umap")
    if isinstance(umap_meta, dict) and umap_meta.get("path"):
        umap_df = pd.read_parquet(umap_meta["path"])

    return {
        "dataframe": loaded_df,
        "umap": umap_df,
        "metadata": dict(artifact_meta),
    }


def load_dataset_embedding_artifacts(
    dataset_meta: dict[str, dict[str, Any]],
    annotation_column: str | None = None,
) -> dict[str, Any]:
    """Load all parquet-backed embedding artifacts for one dataset entry."""
    artifacts: dict[str, dict[str, Any]] = {}

    for artifact_name, artifact_meta in dataset_meta.items():
        if not isinstance(artifact_meta, dict) or "path" not in artifact_meta:
            continue

        artifact_annotation_column = annotation_column if artifact_name == "df_cells" else None
        artifacts[artifact_name] = load_embedding_artifact(
            artifact_meta,
            annotation_column=artifact_annotation_column,
        )

    return {
        "artifacts": artifacts,
        "run_timestamp": infer_embedding_run_timestamp(dataset_meta),
    }

def load_embedding_model(model_name: str):
    from alias.util.load_hf_model import load_model

    return load_model(model_name)


def load_embedding_run_metadata(run_dir: Path | str) -> dict[str, Any]:
    """Load one embedding run's saved artifact metadata."""
    loaded = load_saved_embeddings(run_dir)
    if isinstance(loaded, dict) and len(loaded) == 1:
        return next(iter(loaded.values()))
    if isinstance(loaded, dict):
        return loaded
    raise TypeError("Expected embedding run metadata when loading a run directory.")


def generate_celltype_label_embedding_variant(
    dataset_meta: dict[str, dict[str, Any]],
    model_name: str,
    embedding_config: GenEmbeddingsConfig,
    dataset_name: str,
    label_batch_size: int,
    timestamp: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Regenerate only df_celltypes while reusing the rest of an embedding run."""
    loaded_dataset = load_dataset_embedding_artifacts(
        dataset_meta,
        annotation_column=embedding_config.annotation_column,
    )
    loaded_artifacts = loaded_dataset["artifacts"]
    if "df_celltypes" not in loaded_artifacts:
        raise KeyError("dataset_meta does not contain a df_celltypes embedding artifact.")

    df_celltypes = loaded_artifacts["df_celltypes"]["dataframe"].copy()
    label_column = embedding_config.annotation_column
    if label_column not in df_celltypes.columns:
        label_column = "cell_type"
    if label_column not in df_celltypes.columns:
        raise KeyError(
            "df_celltypes artifact must contain either the annotation column "
            f"`{embedding_config.annotation_column}` or `cell_type`."
        )

    texts = df_celltypes[label_column].astype(str).tolist()
    st_model = load_embedding_model(model_name)
    emb = sentence_transformer_embeddings(
        texts,
        st_model,
        embedding_config,
        batch_size=label_batch_size,
    )

    emb_df = pd.DataFrame(np.array(emb))
    emb_df.index = df_celltypes.index
    emb_df.index.name = "cell_id"

    embeddings_root = resolve_artifact_root(embedding_config.output_dir, "embeddings")
    run_dir = create_run_directory(
        root_dir=embeddings_root,
        category="",
        dataset_name=dataset_name,
        model_name=clean_model_name(model_name),
        evaluation_name=f"df_celltypes_label_batch_{label_batch_size}",
        timestamp=timestamp,
    )
    meta_info = save_embedding_frame(
        run_dir,
        "df_celltypes",
        emb_df,
        annotation_map=df_celltypes[label_column].astype(str).to_dict(),
        annotation_file_name="df_celltypes_annotations.json",
    )
    meta_info.update(
        {
            "dataset": dataset_name,
            "entity_type": "df_celltypes",
            "column": label_column,
            "run_dir": str(run_dir),
            "label_batch_size": label_batch_size,
            "source_embedding_run_timestamp": loaded_dataset["run_timestamp"],
            "source_cell_embedding_run_dir": dataset_meta.get("df_cells", {}).get("run_dir"),
        }
    )

    variant_meta = dict(dataset_meta)
    variant_meta["df_celltypes"] = meta_info
    write_metadata(
        run_dir,
        {
            "dataset_name": dataset_name,
            "model_name": clean_model_name(model_name),
            "annotation_column": embedding_config.annotation_column,
            "run_timestamp": Path(run_dir).name,
            "evaluation_name": "df_celltypes_label_batch_variant",
            "label_batch_size": label_batch_size,
            "source_embedding_run_timestamp": loaded_dataset["run_timestamp"],
            "source_cell_embedding_run_dir": dataset_meta.get("df_cells", {}).get("run_dir"),
            "artifacts": {"df_celltypes": meta_info},
        },
    )
    return variant_meta


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
        cleaned_name = output_model_name(model_name, embedding_config)
        model_metadata = {}
        embeddings_root = resolve_artifact_root(embedding_config.output_dir, "embeddings")

        for dataset_name, dataset_dfs in dfs_dict.items():
            reuse_fields = build_embedding_reuse_config(dataset_name, model_name, embedding_config)
            reuse_signature = build_embedding_reuse_signature(reuse_fields)
            if not embedding_config.force_regenerate:
                reusable_run = find_reusable_embedding_run(
                    embeddings_root=embeddings_root,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    embedding_config=embedding_config,
                )
                if reusable_run is not None:
                    print(f"Reusing embeddings from {reusable_run}")
                    model_metadata[dataset_name] = load_embedding_run_metadata(reusable_run)
                    continue

            st_model = load_embedding_model(model_name)
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
                
                emb = sentence_transformer_embeddings(
                    texts,
                    st_model,
                    embedding_config,
                )
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
                    "reuse_config": reuse_fields,
                    "reuse_signature": reuse_signature,
                    "artifacts": dataset_metadata,
                },
            )
            with (run_dir / "embedding_metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(dataset_metadata, handle, indent=2)

        embeddings_dict[cleaned_name] = model_metadata

    return embeddings_dict
