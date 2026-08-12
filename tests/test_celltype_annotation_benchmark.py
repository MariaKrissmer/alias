from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from alias.evaluation.celltype_annotation import benchmark
from alias.evaluation.celltype_annotation.celltypist import (
    CellTypistModelConfig,
    CellTypistTrainingConfig,
    resolve_celltypist_model,
    train_celltypist_model_from_dataset_dir,
)
from alias.evaluation.celltype_annotation.singler import (
    SingleRAnnotationConfig,
    build_singler_reference_from_dataset_dir,
    run_singler_annotation,
)
from alias.evaluation.celltype_annotation.sctype import (
    ScTypeAnnotationConfig,
    load_sctype_marker_map,
    run_sctype_annotation,
)


def test_annotation_metrics_include_summary_and_per_celltype_scores(tmp_path: Path):
    predictions = pd.DataFrame(
        {
            "true_label": ["T", "T", "B", "B"],
            "predicted_label": ["T", "B", "B", "B"],
        },
        index=pd.Index(["cell1", "cell2", "cell3", "cell4"], name="cell_id"),
    )

    outputs = benchmark.write_annotation_evaluation(
        predictions=predictions,
        output_dir=tmp_path,
        model_name="demo_model",
        dataset_id="demo_dataset",
        metadata={"source": "unit-test"},
    )

    summary = pd.read_csv(outputs["metrics_summary_csv"])
    per_celltype = pd.read_csv(outputs["per_celltype_metrics_csv"])
    confusion = pd.read_csv(outputs["confusion_matrix_csv"], index_col=0)

    assert summary.loc[0, "model_name"] == "demo_model"
    assert summary.loc[0, "dataset_id"] == "demo_dataset"
    assert summary.loc[0, "accuracy"] == 0.75
    assert set(per_celltype["cell_type"]) == {"B", "T"}
    assert confusion.loc["T", "B"] == 1
    assert outputs["confusion_matrix_pdf"].exists()
    assert outputs["confusion_matrix_png"].exists()
    assert outputs["metadata_json"].exists()


def test_annotation_cache_status_distinguishes_complete_partial_and_missing(tmp_path: Path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    assert benchmark.get_annotation_cache_status(model_dir) == "missing_predictions"

    pd.DataFrame({"true_label": ["A"], "predicted_label": ["A"]}).to_csv(
        model_dir / "predictions.csv",
        index=False,
    )
    assert benchmark.get_annotation_cache_status(model_dir) == "missing_evaluation"

    for file_name in benchmark.REQUIRED_EVALUATION_FILES:
        (model_dir / file_name).write_text("placeholder\n", encoding="utf-8")

    assert benchmark.get_annotation_cache_status(model_dir) == "complete"


def test_similarity_predictions_assign_top_label():
    cell_embeddings = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]],
        index=pd.Index(["cell1", "cell2", "cell3"], name="cell_id"),
    )
    cell_embeddings["AIFI_L2"] = ["T", "B", "T"]

    label_embeddings = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
    )
    label_embeddings["cell_type"] = ["T", "B"]

    predictions = benchmark.make_similarity_top_label_predictions(
        cell_embeddings=cell_embeddings,
        label_embeddings=label_embeddings,
        annotation_column="AIFI_L2",
        label_column="cell_type",
    )

    assert predictions.index.tolist() == ["cell1", "cell2", "cell3"]
    assert predictions["true_label"].tolist() == ["T", "B", "T"]
    assert predictions["predicted_label"].tolist()[:2] == ["T", "B"]
    assert predictions["score"].between(-1, 1).all()


def test_celltypist_local_model_resolves_existing_file(tmp_path: Path):
    model_path = tmp_path / "ref_pbmc_clean_celltypist_model_AIFI_L2_2024-04-19.pkl"
    model_path.write_text("placeholder", encoding="utf-8")

    resolved = resolve_celltypist_model(
        CellTypistModelConfig(
            name="AIFI_L2",
            model=model_path,
            source="local",
        ),
        celltypist_module=None,
    )

    assert resolved == str(model_path)


def test_celltypist_builtin_model_downloads_collection_model():
    class FakeModels:
        def __init__(self):
            self.calls = []

        def download_models(self, *, force_update, model):
            self.calls.append({"force_update": force_update, "model": model})

    class FakeCellTypist:
        models = FakeModels()

    resolved = resolve_celltypist_model(
        CellTypistModelConfig(
            name="CellTypist_PBMC",
            model="Healthy_COVID19_PBMC.pkl",
            source="celltypist",
            force_update=True,
        ),
        celltypist_module=FakeCellTypist,
    )

    assert resolved == "Healthy_COVID19_PBMC.pkl"
    assert FakeCellTypist.models.calls == [
        {"force_update": True, "model": ["Healthy_COVID19_PBMC.pkl"]}
    ]


def test_celltypist_training_uses_matching_train_indices_and_labels(tmp_path: Path):
    source_path = tmp_path / "source.h5ad"
    dataset_dir = tmp_path / "S2_heldout_donor_semantic"
    metadata_dir = dataset_dir / "metadata"
    model_path = dataset_dir / "celltype_annotation" / "models" / "CellTypist_train.pkl"
    metadata_dir.mkdir(parents=True)

    adata = ad.AnnData(
        X=np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 2.0],
            ],
            dtype=np.float32,
        )
    )
    adata.obs_names = ["train_a", "test_a", "train_b"]
    adata.var_names = ["Gene1", "Gene2"]
    adata.obs["AIFI_L2"] = ["T cell", "B cell", "Monocyte"]
    adata.write_h5ad(source_path)

    (metadata_dir / "generation_metadata.json").write_text(
        """{
  "source": "%s",
  "scrna_config": {
    "annotation_column": "AIFI_L2",
    "preprocessing": false,
    "highly_variable_genes": false,
    "housekeeping_genes": true,
    "dataset_id": "S2_heldout_donor_semantic"
  }
}
"""
        % source_path,
        encoding="utf-8",
    )
    (metadata_dir / "split_indices.json").write_text(
        """{
  "train_indices": ["train_a", "train_b"],
  "test_indices": ["test_a"]
}
""",
        encoding="utf-8",
    )

    calls = []

    class FakeModel:
        def write(self, path):
            Path(path).write_text("fake model", encoding="utf-8")

    class FakeCellTypist:
        @staticmethod
        def train(**kwargs):
            calls.append(kwargs)
            return FakeModel()

    resolved_model_path = train_celltypist_model_from_dataset_dir(
        CellTypistTrainingConfig(
            dataset_dir=dataset_dir,
            annotation_column="AIFI_L2",
            model_path=model_path,
            model_name="CellTypist_train",
            max_iter=500,
            n_jobs=2,
            force_retrain=True,
        ),
        celltypist_module=FakeCellTypist,
    )

    assert resolved_model_path == model_path
    assert model_path.exists()
    assert len(calls) == 1
    trained_adata = calls[0]["X"]
    assert trained_adata.obs_names.tolist() == ["train_a", "train_b"]
    assert calls[0]["labels"].tolist() == ["T cell", "Monocyte"]
    assert calls[0]["max_iter"] == 500
    assert calls[0]["n_jobs"] == 2


def test_celltypist_training_reuses_matching_cached_model(tmp_path: Path):
    source_path = tmp_path / "source.h5ad"
    dataset_dir = tmp_path / "S2_heldout_donor_semantic"
    metadata_dir = dataset_dir / "metadata"
    model_path = dataset_dir / "celltype_annotation" / "models" / "CellTypist_train.pkl"
    metadata_dir.mkdir(parents=True)
    model_path.parent.mkdir(parents=True)

    adata = ad.AnnData(X=np.array([[1.0], [2.0]], dtype=np.float32))
    adata.obs_names = ["train_a", "test_a"]
    adata.var_names = ["Gene1"]
    adata.obs["AIFI_L2"] = ["T cell", "B cell"]
    adata.write_h5ad(source_path)

    (metadata_dir / "generation_metadata.json").write_text(
        """{
  "source": "%s",
  "scrna_config": {
    "annotation_column": "AIFI_L2",
    "preprocessing": false,
    "highly_variable_genes": false,
    "housekeeping_genes": true
  }
}
"""
        % source_path,
        encoding="utf-8",
    )
    (metadata_dir / "split_indices.json").write_text(
        """{
  "train_indices": ["train_a"],
  "test_indices": ["test_a"]
}
""",
        encoding="utf-8",
    )

    class FakeModel:
        def write(self, path):
            Path(path).write_text("fake model", encoding="utf-8")

    class FakeCellTypist:
        calls = 0

        @classmethod
        def train(cls, **kwargs):
            cls.calls += 1
            return FakeModel()

    config = CellTypistTrainingConfig(
        dataset_dir=dataset_dir,
        annotation_column="AIFI_L2",
        model_path=model_path,
        model_name="CellTypist_train",
        max_iter=500,
    )
    train_celltypist_model_from_dataset_dir(config, celltypist_module=FakeCellTypist)
    train_celltypist_model_from_dataset_dir(config, celltypist_module=FakeCellTypist)

    assert FakeCellTypist.calls == 1


def test_singler_reference_uses_matching_train_indices(tmp_path: Path):
    source_path = tmp_path / "source.h5ad"
    dataset_dir = tmp_path / "S2_heldout_donor_semantic"
    metadata_dir = dataset_dir / "metadata"
    reference_dir = dataset_dir / "celltype_annotation" / "references"
    metadata_dir.mkdir(parents=True)

    adata = ad.AnnData(
        X=np.array(
            [
                [1.0, 0.0, 2.0],
                [0.0, 3.0, 1.0],
                [2.0, 1.0, 0.0],
                [5.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    )
    adata.obs_names = ["train_a", "test_a", "train_b", "unused"]
    adata.var_names = ["Gene1", "Gene2", "Gene3"]
    adata.obs["AIFI_L2"] = ["T cell", "B cell", "Monocyte", "NK cell"]
    adata.write_h5ad(source_path)

    (metadata_dir / "generation_metadata.json").write_text(
        """{
  "source": "%s",
  "scrna_config": {
    "annotation_column": "AIFI_L2",
    "preprocessing": false,
    "highly_variable_genes": false,
    "housekeeping_genes": true,
    "dataset_id": "S2_heldout_donor_semantic"
  }
}
"""
        % source_path,
        encoding="utf-8",
    )
    (metadata_dir / "split_indices.json").write_text(
        """{
  "train_indices": ["train_a", "train_b"],
  "test_indices": ["test_a"]
}
""",
        encoding="utf-8",
    )

    reference = build_singler_reference_from_dataset_dir(
        dataset_dir=dataset_dir,
        annotation_column="AIFI_L2",
        reference_cache_dir=reference_dir,
        force_rebuild=True,
    )

    assert reference.obs_names.tolist() == ["train_a", "train_b"]
    assert reference.obs["AIFI_L2"].tolist() == ["T cell", "Monocyte"]
    assert reference.var_names.tolist() == ["Gene1", "Gene2", "Gene3"]
    assert (reference_dir / "adata_train_reference.h5ad").exists()


def test_singler_reference_can_use_raw_with_obs_value_mapping(tmp_path: Path):
    source_path = tmp_path / "source.h5ad"
    dataset_dir = tmp_path / "S2_heldout_donor_semantic"
    metadata_dir = dataset_dir / "metadata"
    reference_dir = dataset_dir / "celltype_annotation" / "references"
    metadata_dir.mkdir(parents=True)

    adata = ad.AnnData(
        X=np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        )
    )
    adata.obs_names = ["train_a", "test_a"]
    adata.var_names = ["LoggedGene1", "LoggedGene2"]
    adata.obs["AIFI_L2"] = ["gdT", "B cell"]

    raw = ad.AnnData(
        X=np.array(
            [
                [10.0, 2.0, 0.0],
                [0.0, 3.0, 5.0],
            ],
            dtype=np.float32,
        ),
        obs=adata.obs.copy(),
    )
    raw.obs_names = adata.obs_names.copy()
    raw.var_names = ["RawGene1", "RawGene2", "RawGene3"]
    adata.raw = raw
    adata.write_h5ad(source_path)

    (metadata_dir / "generation_metadata.json").write_text(
        """{
  "source": "%s",
  "scrna_config": {
    "annotation_column": "AIFI_L2",
    "preprocessing": false,
    "highly_variable_genes": false,
    "housekeeping_genes": true,
    "dataset_id": "S2_heldout_donor_semantic"
  }
}
"""
        % source_path,
        encoding="utf-8",
    )
    (metadata_dir / "split_indices.json").write_text(
        """{
  "train_indices": ["train_a"],
  "test_indices": ["test_a"]
}
""",
        encoding="utf-8",
    )

    reference = build_singler_reference_from_dataset_dir(
        dataset_dir=dataset_dir,
        annotation_column="AIFI_L2",
        reference_cache_dir=reference_dir,
        force_rebuild=True,
        use_raw=True,
        obs_value_map={"AIFI_L2": {"gdT": "gamma delta T"}},
    )

    assert reference.obs_names.tolist() == ["train_a"]
    assert reference.var_names.tolist() == ["RawGene1", "RawGene2", "RawGene3"]
    assert reference.obs["AIFI_L2"].tolist() == ["gamma delta T"]


def test_singler_annotation_aligns_genes_and_standardizes_predictions(tmp_path: Path):
    query_path = tmp_path / "query.h5ad"
    reference_path = tmp_path / "reference.h5ad"

    query = ad.AnnData(X=np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]], dtype=np.float32))
    query.obs_names = ["query_a", "query_b"]
    query.var_names = ["Gene2", "Gene1", "Gene3"]
    query.obs["AIFI_L2"] = ["T cell", "B cell"]
    query.write_h5ad(query_path)

    reference = ad.AnnData(X=np.array([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32))
    reference.obs_names = ["ref_a", "ref_b"]
    reference.var_names = ["Gene1", "Gene2"]
    reference.obs["AIFI_L2"] = ["T cell", "B cell"]
    reference.write_h5ad(reference_path)

    calls = {}

    class FakeSingleR:
        @staticmethod
        def annotate_single(**kwargs):
            calls.update(kwargs)
            return {
                "best": ["T cell", "B cell"],
                "scores": pd.DataFrame(
                    {
                        "T cell": [0.9, 0.2],
                        "B cell": [0.1, 0.8],
                    }
                ),
            }

    predictions = run_singler_annotation(
        SingleRAnnotationConfig(
            adata_path=query_path,
            reference_adata_path=reference_path,
            annotation_column="AIFI_L2",
            model_name="SingleR_test",
        ),
        singler_module=FakeSingleR,
    )

    assert calls["test_features"] == ["Gene1", "Gene2"]
    assert calls["ref_features"] == ["Gene1", "Gene2"]
    assert calls["test_data"].shape == (2, 2)
    assert calls["ref_data"].shape == (2, 2)
    assert predictions.index.tolist() == ["query_a", "query_b"]
    assert predictions["true_label"].tolist() == ["T cell", "B cell"]
    assert predictions["predicted_label"].tolist() == ["T cell", "B cell"]
    assert predictions["score"].tolist() == [0.9, 0.8]
    assert predictions["model_name"].tolist() == ["SingleR_test", "SingleR_test"]


def test_singler_annotation_caps_reference_cells_per_label(tmp_path: Path):
    query_path = tmp_path / "query.h5ad"
    reference_path = tmp_path / "reference.h5ad"

    query = ad.AnnData(X=np.array([[1.0, 2.0]], dtype=np.float32))
    query.obs_names = ["query_a"]
    query.var_names = ["Gene1", "Gene2"]
    query.obs["AIFI_L2"] = ["T cell"]
    query.write_h5ad(query_path)

    reference = ad.AnnData(
        X=np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [0.0, 4.0],
            ],
            dtype=np.float32,
        )
    )
    reference.obs_names = ["ref_t1", "ref_t2", "ref_t3", "ref_b1"]
    reference.var_names = ["Gene1", "Gene2"]
    reference.obs["AIFI_L2"] = ["T cell", "T cell", "T cell", "B cell"]
    reference.write_h5ad(reference_path)

    calls = {}

    class FakeSingleR:
        @staticmethod
        def annotate_single(**kwargs):
            calls.update(kwargs)
            return {"best": ["T cell"]}

    run_singler_annotation(
        SingleRAnnotationConfig(
            adata_path=query_path,
            reference_adata_path=reference_path,
            annotation_column="AIFI_L2",
            model_name="SingleR_test",
            max_reference_cells_per_label=2,
            reference_sample_seed=42,
        ),
        singler_module=FakeSingleR,
    )

    assert calls["ref_data"].shape == (2, 3)
    assert pd.Series(calls["ref_labels"]).value_counts().to_dict() == {
        "T cell": 2,
        "B cell": 1,
    }


def test_sctype_annotation_scores_marker_sets_and_standardizes_predictions(tmp_path: Path):
    query_path = tmp_path / "query.h5ad"
    query = ad.AnnData(
        X=np.array(
            [
                [8.0, 0.0, 0.0],
                [0.0, 7.0, 0.0],
                [1.0, 1.0, 5.0],
            ],
            dtype=np.float32,
        )
    )
    query.obs_names = ["cell_t", "cell_b", "cell_mono"]
    query.var_names = ["CD3D", "MS4A1", "LYZ"]
    query.obs["AIFI_L2"] = ["T cell", "B cell", "Monocyte"]
    query.write_h5ad(query_path)

    predictions = run_sctype_annotation(
        ScTypeAnnotationConfig(
            adata_path=query_path,
            annotation_column="AIFI_L2",
            model_name="scType_test",
            marker_map={
                "T cell": {"positive": ["CD3D"]},
                "B cell": {"positive": ["MS4A1"]},
                "Monocyte": {"positive": ["LYZ"]},
            },
            scale_expression=False,
        )
    )

    assert predictions.index.tolist() == ["cell_t", "cell_b", "cell_mono"]
    assert predictions["true_label"].tolist() == ["T cell", "B cell", "Monocyte"]
    assert predictions["predicted_label"].tolist() == ["T cell", "B cell", "Monocyte"]
    assert predictions["model_name"].tolist() == ["scType_test"] * 3
    assert predictions["n_present_positive_markers"].tolist() == [1, 1, 1]


def test_sctype_marker_csv_parser_accepts_positive_and_negative_markers(tmp_path: Path):
    marker_path = tmp_path / "markers.csv"
    marker_path.write_text(
        "cell_type,positive_markers,negative_markers\n"
        "T cell,CD3D;TRAC,MS4A1\n"
        "B cell,MS4A1|CD79A,CD3D\n",
        encoding="utf-8",
    )

    marker_map = load_sctype_marker_map(marker_path)

    assert marker_map == {
        "T cell": {"positive": ["CD3D", "TRAC"], "negative": ["MS4A1"]},
        "B cell": {"positive": ["MS4A1", "CD79A"], "negative": ["CD3D"]},
    }
