from pathlib import Path

import json
import pandas as pd

from alias.util.artifacts import (
    create_evaluation_run_directory,
    create_run_directory,
    load_annotation_map,
    load_embedding_frame,
    save_embedding_frame,
    write_metadata,
)


def test_create_run_directory_creates_timestamped_hierarchy(tmp_path: Path):
    run_dir = create_run_directory(
        root_dir=tmp_path,
        category="embeddings",
        dataset_name="pbmc_3k",
        model_name="pubmedbert/base",
        evaluation_name="celltype labels",
        timestamp="2026-03-23T12-34-56",
    )

    assert run_dir.exists()
    assert run_dir == tmp_path / "embeddings" / "pbmc_3k" / "pubmedbertbase" / "celltype_labels" / "2026-03-23T12-34-56"


def test_write_metadata_writes_json(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    metadata = {"dataset": "pbmc_3k", "n_cells": 10}
    metadata_path = write_metadata(run_dir, metadata)

    assert metadata_path == run_dir / "metadata.json"
    assert metadata_path.exists()
    with metadata_path.open() as handle:
        assert json.load(handle) == metadata


def test_create_run_directory_appends_suffix_when_timestamp_collides(tmp_path: Path):
    first_run_dir = create_run_directory(
        root_dir=tmp_path,
        category="embeddings",
        dataset_name="pbmc_3k",
        model_name="pubmedbert/base",
        timestamp="2026-03-23T12-34-56",
    )

    second_run_dir = create_run_directory(
        root_dir=tmp_path,
        category="embeddings",
        dataset_name="pbmc_3k",
        model_name="pubmedbert/base",
        timestamp="2026-03-23T12-34-56",
    )

    assert first_run_dir.exists()
    assert second_run_dir.exists()
    assert second_run_dir != first_run_dir
    assert second_run_dir.name == "2026-03-23T12-34-56_01"


def test_create_evaluation_run_directory_matches_preferred_layout(tmp_path: Path):
    run_dir = create_evaluation_run_directory(
        output_dir=tmp_path,
        model_name="pubmedbert/base",
        dataset_name="pbmc_3k",
        evaluation_name="functionality_similarity",
        timestamp="2026-03-23T12-34-56",
    )

    assert run_dir.exists()
    assert run_dir == (
        tmp_path
        / "pubmedbertbase"
        / "pbmc_3k"
        / "functionality_similarity"
        / "2026-03-23T12-34-56"
    )


def test_save_and_load_embedding_frame_roundtrip(tmp_path: Path):
    embedding_df = pd.DataFrame(
        [[1.0, 0.0], [0.5, 0.5]],
        index=pd.Index(["cell_a", "cell_b"], name="cell_id"),
    )

    metadata = save_embedding_frame(tmp_path, "df_cells", embedding_df)

    loaded_df = load_embedding_frame(metadata["path"])

    pd.testing.assert_frame_equal(loaded_df, embedding_df)
    assert metadata["path"].endswith("df_cells.parquet")
    assert metadata["n_samples"] == 2
    assert metadata["embedding_dim"] == 2


def test_save_embedding_frame_persists_annotation_map(tmp_path: Path):
    embedding_df = pd.DataFrame(
        [[1.0, 0.0]],
        index=pd.Index(["cell_a"], name="cell_id"),
    )
    annotation_map = {"celltype": {"cell_a": "T_cell"}}

    metadata = save_embedding_frame(
        tmp_path,
        "df_cells",
        embedding_df,
        annotation_map=annotation_map,
        annotation_file_name="df_cells_annotations.json",
    )

    assert metadata["annotation_map"].endswith("df_cells_annotations.json")
    assert load_annotation_map(metadata["annotation_map"]) == annotation_map
