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
load_dataset_embedding_artifacts = embedding_module.load_dataset_embedding_artifacts
generate_celltype_label_embedding_variant = embedding_module.generate_celltype_label_embedding_variant


class DummySentenceTransformer:
    def encode(self, texts, batch_size=64, show_progress_bar=True):
        return [[float(len(text)), float(index)] for index, text in enumerate(texts)]

    def get_sentence_embedding_dimension(self):
        return 2


class RecordingSentenceTransformer:
    def __init__(self):
        self.calls = []

    def encode(self, texts, batch_size=64, show_progress_bar=True):
        text_list = list(texts)
        self.calls.append({"texts": text_list, "batch_size": batch_size})
        return [[float(len(text)), float(index)] for index, text in enumerate(text_list)]

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


def test_load_dataset_embedding_artifacts_restores_frames_umaps_and_metadata(tmp_path: Path):
    run_dir = tmp_path / "embeddings" / "scrna" / "demo_model" / "2026-03-31T10-00-00"
    run_dir.mkdir(parents=True)

    cell_df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=pd.Index(["0", "1"], name="cell_id"))
    additional_df = pd.DataFrame([[5.0, 6.0]], index=pd.Index(["f0"], name="cell_id"))

    cell_meta = save_embedding_frame(
        run_dir,
        "df_cells",
        cell_df,
        annotation_map={"celltype": {"0": "T_cell", "1": "B_cell"}},
    )
    cell_meta["run_dir"] = str(run_dir)

    additional_meta = save_embedding_frame(
        run_dir,
        "df_additional",
        additional_df,
        annotation_map={"f0": "first functionality"},
        annotation_file_name="df_additional_input_mapping.json",
    )
    additional_meta["run_dir"] = str(run_dir)

    umap_dir = run_dir / "umap"
    umap_dir.mkdir()
    cell_umap = pd.DataFrame(
        {"UMAP1": [0.0, 1.0], "UMAP2": [1.0, 2.0]},
        index=pd.Index(["0", "1"], name="cell_id"),
    )
    cell_umap_path = umap_dir / "df_cells_umap.parquet"
    cell_umap.to_parquet(cell_umap_path)
    cell_meta["umap"] = {"path": str(cell_umap_path), "n_points": 2}

    loaded = load_dataset_embedding_artifacts(
        {
            "df_cells": cell_meta,
            "df_additional": additional_meta,
        },
        annotation_column="celltype",
    )

    assert loaded["run_timestamp"] == "2026-03-31T10-00-00"
    assert list(loaded["artifacts"]["df_cells"]["dataframe"]["celltype"]) == ["T_cell", "B_cell"]
    assert loaded["artifacts"]["df_cells"]["metadata"]["path"] == cell_meta["path"]
    pd.testing.assert_frame_equal(loaded["artifacts"]["df_cells"]["umap"], cell_umap)

    additional_loaded = loaded["artifacts"]["df_additional"]["dataframe"]
    assert list(additional_loaded["data"]) == ["first functionality"]
    assert loaded["artifacts"]["df_additional"]["umap"] is None


def test_generate_embeddings_reuses_matching_config_run(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        embedding_module,
        "load_embedding_model",
        lambda _: DummySentenceTransformer(),
    )

    evaluation_dict = {
        "scrna": {
            "test": [
                {"index": "cell_alpha", "sentence1": "GENE1 GENE2", "celltype": "T_cell", "label": "T_cell"},
                {"index": "cell_beta", "sentence1": "GENE3 GENE4", "celltype": "B_cell", "label": "B_cell"},
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

    first = generate_embeddings(
        evaluation_dict,
        config,
        timestamp="2026-04-01T10-00-00",
    )
    second = generate_embeddings(
        evaluation_dict,
        config,
        timestamp="2026-04-01T11-00-00",
    )

    first_path = first["modelname"]["scrna"]["df_cells"]["path"]
    second_path = second["modelname"]["scrna"]["df_cells"]["path"]
    assert second_path == first_path


def test_generate_embeddings_uses_configured_output_model_name_for_checkpoint_paths(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.setattr(
        embedding_module,
        "load_embedding_model",
        lambda _: DummySentenceTransformer(),
    )

    checkpoint_path = "/models/MO_LaManno_S9_N5_lr5e5_all/epoch_11_ncbi"
    evaluation_dict = {
        "scrna": {
            "test": [
                {"index": "cell_alpha", "sentence1": "GENE1 GENE2", "celltype": "Forebrain", "label": "Forebrain"},
            ]
        }
    }
    config = GenEmbeddingsConfig(
        annotation_column="celltype",
        embedding_models=[checkpoint_path],
        model_type="sentence_transformer",
        output_dir=str(tmp_path / "_out"),
        max_cells=10,
        model_output_names={checkpoint_path: "MO_LaManno_S9_N5_lr5e5_epoch_11_ncbi"},
    )

    embeddings = generate_embeddings(
        evaluation_dict,
        config,
        timestamp="2026-04-01T10-00-00",
    )

    assert list(embeddings) == ["MO_LaManno_S9_N5_lr5e5_epoch_11_ncbi"]
    cell_path = Path(
        embeddings["MO_LaManno_S9_N5_lr5e5_epoch_11_ncbi"]["scrna"]["df_cells"]["path"]
    )
    assert cell_path.parent.parent.name == "MO_LaManno_S9_N5_lr5e5_epoch_11_ncbi"


def test_generate_embeddings_creates_new_run_when_config_differs(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        embedding_module,
        "load_embedding_model",
        lambda _: DummySentenceTransformer(),
    )

    evaluation_dict = {
        "scrna": {
            "test": [
                {"index": "cell_alpha", "sentence1": "GENE1 GENE2", "celltype": "T_cell", "label": "T_cell"},
                {"index": "cell_beta", "sentence1": "GENE3 GENE4", "celltype": "B_cell", "label": "B_cell"},
            ]
        }
    }
    base_kwargs = dict(
        annotation_column="celltype",
        embedding_models=["org/model-name"],
        model_type="sentence_transformer",
        output_dir=str(tmp_path / "_out"),
    )

    first = generate_embeddings(
        evaluation_dict,
        GenEmbeddingsConfig(**base_kwargs, max_cells=10),
        timestamp="2026-04-01T10-00-00",
    )
    second = generate_embeddings(
        evaluation_dict,
        GenEmbeddingsConfig(**base_kwargs, max_cells=1),
        timestamp="2026-04-01T11-00-00",
    )

    assert second["modelname"]["scrna"]["df_cells"]["path"] != first["modelname"]["scrna"]["df_cells"]["path"]


def test_generate_embeddings_uses_configured_batch_size_for_additional_data(tmp_path: Path, monkeypatch):
    model = RecordingSentenceTransformer()
    monkeypatch.setattr(
        embedding_module,
        "load_embedding_model",
        lambda _: model,
    )

    evaluation_dict = {
        "scrna": {
            "test": [
                {"index": "cell_alpha", "sentence1": "GENE1 GENE2", "celltype": "T_cell", "label": "T_cell"},
                {"index": "cell_beta", "sentence1": "GENE3 GENE4", "celltype": "B_cell", "label": "B_cell"},
            ]
        }
    }
    config = GenEmbeddingsConfig(
        annotation_column="celltype",
        embedding_models=["org/model-name"],
        model_type="sentence_transformer",
        output_dir=str(tmp_path / "_out"),
        batch_size=7,
        max_cells=10,
        additional_data=["cytotoxic", "regulatory"],
    )

    generate_embeddings(
        evaluation_dict,
        config,
        timestamp="2026-04-01T12-00-00",
    )

    additional_call = next(
        call for call in model.calls if call["texts"] == ["cytotoxic", "regulatory"]
    )
    non_additional_batch_sizes = [
        call["batch_size"]
        for call in model.calls
        if call["texts"] != ["cytotoxic", "regulatory"]
    ]

    assert additional_call["batch_size"] == 7
    assert set(non_additional_batch_sizes) == {7}


def test_generate_embeddings_force_regenerate_bypasses_reuse(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        embedding_module,
        "load_embedding_model",
        lambda _: DummySentenceTransformer(),
    )

    evaluation_dict = {
        "scrna": {
            "test": [
                {"index": "cell_alpha", "sentence1": "GENE1 GENE2", "celltype": "T_cell", "label": "T_cell"},
                {"index": "cell_beta", "sentence1": "GENE3 GENE4", "celltype": "B_cell", "label": "B_cell"},
            ]
        }
    }
    base_kwargs = dict(
        annotation_column="celltype",
        embedding_models=["org/model-name"],
        model_type="sentence_transformer",
        output_dir=str(tmp_path / "_out"),
        max_cells=10,
    )

    first = generate_embeddings(
        evaluation_dict,
        GenEmbeddingsConfig(**base_kwargs),
        timestamp="2026-04-01T10-00-00",
    )
    second = generate_embeddings(
        evaluation_dict,
        GenEmbeddingsConfig(**base_kwargs, force_regenerate=True),
        timestamp="2026-04-01T11-00-00",
    )

    assert second["modelname"]["scrna"]["df_cells"]["path"] != first["modelname"]["scrna"]["df_cells"]["path"]


def test_generate_celltype_label_embedding_variant_reuses_cells_and_batches_labels(
    tmp_path: Path,
    monkeypatch,
):
    model = RecordingSentenceTransformer()
    monkeypatch.setattr(
        embedding_module,
        "load_embedding_model",
        lambda _: model,
    )

    evaluation_dict = {
        "scrna": {
            "test": [
                {"index": "cell_alpha", "sentence1": "GENE1 GENE2", "celltype": "T_cell", "label": "T_cell"},
                {"index": "cell_beta", "sentence1": "GENE3 GENE4", "celltype": "B_cell", "label": "B_cell"},
            ]
        }
    }
    config = GenEmbeddingsConfig(
        annotation_column="celltype",
        embedding_models=["org/model-name"],
        model_type="sentence_transformer",
        output_dir=str(tmp_path / "_out"),
        batch_size=7,
        max_cells=10,
    )
    original = generate_embeddings(
        evaluation_dict,
        config,
        timestamp="2026-04-01T10-00-00",
    )["modelname"]["scrna"]
    model.calls.clear()

    variant = generate_celltype_label_embedding_variant(
        dataset_meta=original,
        model_name="org/model-name",
        embedding_config=config,
        dataset_name="scrna",
        label_batch_size=3,
        timestamp="2026-04-01T11-00-00",
    )

    assert variant["df_cells"] == original["df_cells"]
    assert variant["df_genes"] == original["df_genes"]
    assert variant["df_labels"] == original["df_labels"]
    assert variant["df_celltypes"]["path"] != original["df_celltypes"]["path"]
    assert variant["df_celltypes"]["label_batch_size"] == 3
    assert model.calls == [{"texts": ["T_cell", "B_cell"], "batch_size": 3}]
