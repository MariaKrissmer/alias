"""Load a locally saved Hugging Face dataset across datasets versions."""

from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk


def load_saved_dataset(path: str | Path) -> Dataset:
    """Read a saved dataset, falling back to its Arrow shards if needed."""
    dataset_path = Path(path)
    try:
        dataset = load_from_disk(str(dataset_path))
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Expected a Dataset at {dataset_path}, got {type(dataset).__name__}")
        return dataset
    except Exception as load_error:
        arrow_files = sorted(dataset_path.glob("*.arrow"))
        if not arrow_files:
            raise RuntimeError(f"Could not load saved dataset at {dataset_path}: {load_error}") from load_error

        import pyarrow as pa
        import pyarrow.ipc as ipc

        shards = []
        for arrow_path in arrow_files:
            with pa.memory_map(str(arrow_path), "r") as source:
                table = ipc.open_stream(source).read_all()
            shards.append(Dataset.from_pandas(table.to_pandas(), preserve_index=False))
        return shards[0] if len(shards) == 1 else concatenate_datasets(shards)
