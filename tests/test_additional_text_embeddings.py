from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

from alias.evaluation import embedding
from alias.util.artifacts import save_embedding_frame


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "revision1_v1" / "HIAI_Tcells" / "celltype_annotation"
sys.path.insert(0, str(SCRIPT_DIR))
import celltype_annotation  # noqa: E402
import celltype_annotation_synonym  # noqa: E402
import _plotting_common  # noqa: E402
import run_celltype_annotation  # noqa: E402
import plot_ablation_annotation_effect  # noqa: E402


def test_additional_text_embeddings_reuse_cells_and_encoded_texts(tmp_path, monkeypatch):
    cell_path = tmp_path / "df_cells.parquet"
    pd.DataFrame({0: [1.0, 2.0]}, index=["cell-1", "cell-2"]).to_parquet(cell_path)
    base = {
        "ALIASMA": {
            "scrna": {
                "df_cells": {
                    "path": str(cell_path),
                    "embedding_dim": 2,
                    "n_samples": 2,
                }
            }
        }
    }

    calls = []

    class FakeModel:
        def encode(self, texts, **kwargs):
            calls.append(list(texts))
            return np.array([[float(i), float(i + 1)] for i in range(len(texts))])

    monkeypatch.setattr(embedding, "load_embedding_model", lambda _: FakeModel())
    kwargs = {
        "embeddings_dict": base,
        "model_name": "mariakrissmer/ALIAS-MA",
        "dataset_name": "scrna",
        "texts": ["T cell", "cytotoxic T cell"],
        "output_dir": tmp_path / "embeddings",
        "batch_size": 8,
    }
    first = embedding.add_additional_text_embeddings(**kwargs)
    second = embedding.add_additional_text_embeddings(**kwargs)

    assert calls == [["T cell", "cytotoxic T cell"]]
    assert first["ALIASMA"]["scrna"]["df_cells"]["path"] == str(cell_path)
    assert second["ALIASMA"]["scrna"]["df_additional"]["path"] == first["ALIASMA"]["scrna"]["df_additional"]["path"]
    assert Path(first["ALIASMA"]["scrna"]["df_additional"]["path"]).is_file()
    loaded = embedding.load_dataset_embedding_artifacts(first["ALIASMA"]["scrna"])
    assert loaded["artifacts"]["df_additional"]["dataframe"]["data"].tolist() == kwargs["texts"]


def test_normal_annotation_uses_precomputed_embeddings(tmp_path, monkeypatch):
    cell_artifact = save_embedding_frame(
        tmp_path / "cells",
        "df_cells",
        pd.DataFrame({0: [1.0, 0.0], 1: [0.0, 1.0]}, index=["cell-1", "cell-2"]),
        annotation_map={"AIFI_L2": {"cell-1": "T", "cell-2": "B"}},
    )
    label_artifact = save_embedding_frame(
        tmp_path / "labels",
        "df_celltypes",
        pd.DataFrame({0: [1.0, 0.0], 1: [0.0, 1.0]}),
        annotation_map={0: "T", 1: "B"},
    )
    precomputed = {
        "ALIASMA": {"scrna": {"df_cells": cell_artifact, "df_celltypes": label_artifact}}
    }
    monkeypatch.setattr(celltype_annotation, "OUTPUT_DIR", tmp_path / "annotation")
    monkeypatch.setattr(
        celltype_annotation.ev,
        "generate_embeddings",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Cells were re-encoded")),
    )
    summary = celltype_annotation.run(model_names=["MB"], embeddings_dict=precomputed)
    predictions = pd.read_csv(tmp_path / "annotation" / "MB" / "predictions.csv")

    assert predictions["predicted_label"].tolist() == ["T", "B"]
    assert summary.iloc[0]["balanced_accuracy"] == 1.0
    assert (tmp_path / "annotation" / "MB" / "celltype_label_similarity.csv").is_file()


def test_synonym_panel_uses_only_new_label_embeddings(tmp_path, monkeypatch):
    mapping = pd.DataFrame(
        [
            ("T", "T", True, "test"),
            ("T", "T lymphocyte", False, "test"),
            ("T", "T-cell", False, "test"),
            ("B", "B", True, "test"),
            ("B", "B lymphocyte", False, "test"),
            ("B", "B-cell", False, "test"),
        ],
        columns=["canonical_label", "synonym_label", "is_canonical_label", "synonym_source"],
    )
    mapping_path = tmp_path / "synonym_mapping.csv"
    mapping.to_csv(mapping_path, index=False)
    cell_artifact = save_embedding_frame(
        tmp_path / "cells",
        "df_cells",
        pd.DataFrame({0: [1.0, 0.0], 1: [0.0, 1.0]}, index=["cell-1", "cell-2"]),
        annotation_map={"AIFI_L2": {"cell-1": "T", "cell-2": "B"}},
    )
    precomputed = {"ALIASMA": {"scrna": {"df_cells": cell_artifact}}}
    annotation_dir = tmp_path / "annotation"
    (annotation_dir / "MB").mkdir(parents=True)
    pd.DataFrame(
        [{"accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0, "weighted_f1": 1.0}]
    ).to_csv(annotation_dir / "MB" / "metrics_summary.csv", index=False)
    monkeypatch.setattr(_plotting_common, "ANNOTATION_DIR", annotation_dir)
    monkeypatch.setattr(celltype_annotation_synonym, "OUTPUT_DIR", tmp_path / "synonym")
    monkeypatch.setattr(celltype_annotation_synonym, "PLOT_OUTPUT_DIR", tmp_path / "plots")
    monkeypatch.setattr(celltype_annotation_synonym, "EMBEDDING_ROOT", tmp_path / "embeddings")
    monkeypatch.setattr(celltype_annotation_synonym, "SYNONYM_MAPPING_PATH", mapping_path)
    calls = []

    class FakeModel:
        def encode(self, texts, **kwargs):
            calls.append(list(texts))
            return np.array(
                [[1.0, 0.0] if text.startswith("T") else [0.0, 1.0] for text in texts]
            )

    monkeypatch.setattr(embedding, "load_embedding_model", lambda _: FakeModel())

    result = celltype_annotation_synonym.run_balanced_accuracy_from_embeddings(
        embeddings_dict=precomputed,
        model_names=["MB"],
        observed_labels={"T", "B"},
    )

    assert len(calls) == 1
    assert set(calls[0]) == set(mapping["synonym_label"])
    assert result.is_file()
    assert (tmp_path / "synonym" / "synonym_metric_comparison.csv").is_file()


def test_figure2c_manifest_accepts_only_embeddings_from_its_run(tmp_path):
    run_dir = tmp_path / "figure2-run"
    cell_path = run_dir / "embeddings" / "scrna" / "ALIASMA" / "run-1" / "df_cells.parquet"
    cell_path.parent.mkdir(parents=True)
    cell_path.write_bytes(b"saved embedding artifact")
    label_path = cell_path.parent / "df_celltypes.parquet"
    label_path.write_bytes(b"saved label artifact")
    manifest_path = run_dir / "embedding_manifest.json"
    payload = {
        "dataset_id": "S2_heldout_donor_semantic_200k",
        "analysis_run": str(run_dir),
        "model_sources": {"MB": "mariakrissmer/ALIAS-MA"},
        "observed_labels": ["T"],
        "embeddings": {
            "ALIASMA": {"scrna": {
                "df_cells": {"path": str(cell_path)},
                "df_celltypes": {"path": str(label_path)},
            }}
        },
    }
    manifest_path.write_text(json.dumps(payload))
    loaded = run_celltype_annotation.load_figure2c_embedding_manifest(
        manifest_path, ["MB"]
    )
    assert loaded["embeddings"]["ALIASMA"]["scrna"]["df_cells"]["path"] == str(cell_path)

    outside_path = tmp_path / "old-embeddings.parquet"
    outside_path.write_bytes(b"old artifact")
    payload["embeddings"]["ALIASMA"]["scrna"]["df_cells"]["path"] = str(outside_path)
    manifest_path.write_text(json.dumps(payload))
    import pytest

    with pytest.raises(ValueError, match="outside the Figure 2 analysis run"):
        run_celltype_annotation.load_figure2c_embedding_manifest(manifest_path, ["MB"])


def test_ablation_plot_reads_new_base_annotation_outputs(tmp_path, monkeypatch):
    base_dir = tmp_path / "annotation" / "Base"
    base_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"accuracy": 0.8, "balanced_accuracy": 0.7, "macro_f1": 0.75, "weighted_f1": 0.76}]
    ).to_csv(base_dir / "metrics_summary.csv", index=False)
    pd.DataFrame(
        [{"cell_type": "T", "roc_auc": 0.9}, {"cell_type": "B", "roc_auc": 0.8}]
    ).to_csv(base_dir / "celltype_label_similarity.csv", index=False)
    monkeypatch.setattr(_plotting_common, "ANNOTATION_DIR", tmp_path / "annotation")
    base_spec = next(
        model for model in plot_ablation_annotation_effect.MODELS if model.label == "Base"
    )

    roc_auc, summary = plot_ablation_annotation_effect.load_ablation_data([base_spec])

    assert roc_auc["plot_model"].tolist() == ["Base", "Base"]
    assert summary.iloc[0]["balanced_accuracy"] == 0.7
