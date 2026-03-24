import importlib.util
from pathlib import Path

import pandas as pd

from alias.util.artifacts import load_annotation_map, load_embedding_frame, save_embedding_frame


EMBEDDING_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "alias" / "evaluation" / "embedding.py"
EMBEDDING_SPEC = importlib.util.spec_from_file_location("alias.evaluation.embedding", EMBEDDING_MODULE_PATH)
embedding_module = importlib.util.module_from_spec(EMBEDDING_SPEC)
assert EMBEDDING_SPEC.loader is not None
EMBEDDING_SPEC.loader.exec_module(embedding_module)

GenEmbeddingsConfig = embedding_module.GenEmbeddingsConfig
generate_embeddings = embedding_module.generate_embeddings
load_saved_embeddings = embedding_module.load_saved_embeddings


class DummySentenceTransformer:
    def encode(self, texts, batch_size=64, show_progress_bar=True):
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]

    def get_sentence_embedding_dimension(self):
        return 2


def test_save_embedding_frame_roundtrip_with_annotation_map(tmp_path: Path):
    embedding_df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["cell_a", "cell_b"])

    metadata = save_embedding_frame(
        output_dir=tmp_path,
        artifact_name="df_cells",
        embedding_df=embedding_df,
        annotation_map={"celltype": {"cell_a": "T", "cell_b": "B"}},
    )

    loaded_df = load_embedding_frame(metadata["path"])
    loaded_map = load_annotation_map(metadata["annotation_map"])

    pd.testing.assert_frame_equal(loaded_df, embedding_df)
    assert loaded_map == {"celltype": {"cell_a": "T", "cell_b": "B"}}


def test_generate_embeddings_creates_timestamped_run_metadata(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        embedding_module,
        "load_embedding_model",
        lambda _: DummySentenceTransformer(),
    )

    evaluation_dict = {
        "scrna": {
            "test": [
                {"sentence1": "GENE1 GENE2", "celltype": "T_cell", "label": "T_cell"},
                {"sentence1": "GENE3 GENE4", "celltype": "B_cell", "label": "B_cell"},
            ]
        }
    }
    config = GenEmbeddingsConfig(
        annotation_column="celltype",
        embedding_models=["org/model-name"],
        model_type="sentence_transformer",
        output_dir=str(tmp_path / "_out"),
        max_cells=10,
    )

    embeddings_dict = generate_embeddings(
        evaluation_dict,
        config,
        timestamp="2026-03-23T12-34-56",
    )

    info = embeddings_dict["modelname"]["scrna"]["df_cells"]

    assert Path(info["path"]).exists()
    assert info["dataset"] == "scrna"
    assert info["entity_type"] == "df_cells"
    assert info["column"] == "sentence1"
    assert info["n_samples"] == 2
    assert info["embedding_dim"] == 2
    assert Path(info["path"]).parts[-5:] == (
        "embeddings",
        "scrna",
        "modelname",
        "2026-03-23T12-34-56",
        "df_cells.parquet",
    )

    run_dir = Path(info["path"]).parent
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "embedding_metadata.json").exists()

    metadata = load_saved_embeddings(run_dir)
    assert metadata["scrna"]["df_cells"]["path"] == info["path"]

    file_metadata = load_saved_embeddings(run_dir / "embedding_metadata.json")
    assert file_metadata["df_cells"]["path"] == info["path"]

    loaded_df = load_saved_embeddings(info["path"], info["annotation_map"])
    assert list(loaded_df["celltype"]) == ["T_cell", "B_cell"]


def test_generate_embeddings_does_not_duplicate_embeddings_directory(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        embedding_module,
        "load_embedding_model",
        lambda _: DummySentenceTransformer(),
    )

    evaluation_dict = {
        "scrna": {
            "test": [
                {"sentence1": "GENE1 GENE2", "celltype": "T_cell", "label": "T_cell"},
            ]
        }
    }
    config = GenEmbeddingsConfig(
        annotation_column="celltype",
        embedding_models=["org/model-name"],
        model_type="sentence_transformer",
        output_dir=str(tmp_path / "embeddings"),
        max_cells=10,
    )

    embeddings_dict = generate_embeddings(
        evaluation_dict,
        config,
        timestamp="2026-03-23T12-34-56",
    )

    info = embeddings_dict["modelname"]["scrna"]["df_cells"]
    assert "embeddings/embeddings" not in info["path"]


def test_load_saved_embeddings_restores_flat_annotation_map(tmp_path: Path):
    embedding_df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["0", "1"])
    metadata = save_embedding_frame(
        output_dir=tmp_path,
        artifact_name="df_celltypes",
        embedding_df=embedding_df,
        annotation_map={"0": "T_cell", "1": "B_cell"},
    )

    loaded_df = load_saved_embeddings(metadata["path"], metadata["annotation_map"])
    assert list(loaded_df["cell_type"]) == ["T_cell", "B_cell"]


def test_generate_embeddings_preserves_source_cell_ids_from_index_column(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        embedding_module,
        "load_embedding_model",
        lambda _: DummySentenceTransformer(),
    )

    evaluation_dict = {
        "scrna": {
            "test": [
                {
                    "index": "cell_alpha",
                    "sentence1": "GENE1 GENE2",
                    "celltype": "T_cell",
                    "label": "T_cell",
                },
                {
                    "index": "cell_beta",
                    "sentence1": "GENE3 GENE4",
                    "celltype": "B_cell",
                    "label": "B_cell",
                },
                {
                    "index": "cell_gamma",
                    "sentence1": "GENE5 GENE6",
                    "celltype": "NK_cell",
                    "label": "NK_cell",
                },
            ]
        }
    }
    config = GenEmbeddingsConfig(
        annotation_column="celltype",
        embedding_models=["org/model-name"],
        model_type="sentence_transformer",
        output_dir=str(tmp_path / "_out"),
        max_cells=2,
    )

    embeddings_dict = generate_embeddings(
        evaluation_dict,
        config,
        timestamp="2026-03-24T12-00-00",
    )

    info = embeddings_dict["modelname"]["scrna"]["df_cells"]
    loaded_df = load_embedding_frame(info["path"])
    loaded_map = load_annotation_map(info["annotation_map"])

    assert set(loaded_df.index) <= {"cell_alpha", "cell_beta", "cell_gamma"}
    assert len(loaded_df.index) == 2
    assert set(loaded_map["celltype"]).issubset({"cell_alpha", "cell_beta", "cell_gamma"})
