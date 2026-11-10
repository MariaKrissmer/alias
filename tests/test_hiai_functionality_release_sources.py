import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from alias.evaluation.embedding import (
    GenEmbeddingsConfig,
    build_embedding_reuse_config,
    build_embedding_reuse_signature,
)
from alias.util.artifacts import save_embedding_frame, write_metadata


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/revision1_v1/HIAI_Tcells/functionality_assessment/run_functionality_benchmark.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("hiai_functionality_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_functionality_generation_reuses_cells_and_encodes_only_missing_text(tmp_path, monkeypatch):
    module = _load_script()
    model = module.ABLATION_MODELS[0]
    monkeypatch.setattr(module, "EMBEDDING_ROOT", tmp_path / "embeddings")
    monkeypatch.setattr(module, "EVALUATION_ROOT", tmp_path / "plots")
    monkeypatch.setattr(module, "ALIAS_RESULTS_ROOT", tmp_path / "scores")
    monkeypatch.setattr(module, "MAX_CELLS", 2)

    cfg = GenEmbeddingsConfig(
        embedding_models=[model.model_id],
        model_type="sentence_transformer",
        batch_size=module.BATCH_SIZE,
        output_dir=str(module.EMBEDDING_ROOT / model.label),
        max_cells=2,
        annotation_column=module.ANNOTATION_COLUMN,
    )
    cache = module.EMBEDDING_ROOT / model.label / "embeddings/scrna/ALIASMA/2026-11-09T09-57-31"
    cache.mkdir(parents=True)
    cell_meta = save_embedding_frame(
        cache,
        "df_cells",
        pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=pd.Index(["1", "2"], name="cell_id")),
        annotation_map={module.ANNOTATION_COLUMN: {"1": "MAIT", "2": "Treg"}},
    )
    cell_meta["run_dir"] = str(cache)
    fields = build_embedding_reuse_config("scrna", model.model_id, cfg)
    write_metadata(
        cache,
        {
            "dataset_name": "scrna",
            "reuse_config": fields,
            "reuse_signature": build_embedding_reuse_signature(fields),
            "artifacts": {"df_cells": cell_meta},
        },
    )

    calls = []

    class TextModel:
        def encode(self, texts, **kwargs):
            calls.append(list(texts))
            return np.ones((len(texts), 2))

    monkeypatch.setattr("alias.evaluation.embedding.load_embedding_model", lambda model_id: TextModel())
    monkeypatch.setattr("alias.evaluation.umap_plots", lambda embeddings_dict, **kwargs: embeddings_dict)

    def fake_functionality_similarity(embeddings_dict, **kwargs):
        artifacts = embeddings_dict[model.label]["scrna"]
        assert artifacts["df_cells"]["path"] == str(cache / "df_cells.parquet")
        assert artifacts["df_additional"]["path"]
        return pd.DataFrame({"functionality": ["test"], "cell_type": ["MAIT"], "mean_auc": [0.8]})

    monkeypatch.setattr("alias.evaluation.functionality_similarity", fake_functionality_similarity)
    first = module._generate_functionality_results(model)
    second = module._generate_functionality_results(model)
    assert first != second
    assert len(calls) == 1
    assert calls[0] == module.CELLTYPE_FUNCTIONALITY_DESCRIPTIONS
    metadata = json.loads((first.parent / "metadata.json").read_text())
    assert metadata["model_id"] == "mariakrissmer/ALIAS-MA"
    assert metadata["source_cell_embedding_run"] == str(cache)
    assert module._latest_functionality_results(model) == second


def test_functionality_generation_fails_without_matching_cell_cache(tmp_path, monkeypatch):
    module = _load_script()
    monkeypatch.setattr(module, "EMBEDDING_ROOT", tmp_path / "embeddings")
    try:
        module._generate_functionality_results(module.ABLATION_MODELS[0])
    except FileNotFoundError as error:
        assert "will not re-encode cells" in str(error)
    else:
        raise AssertionError("Missing cell cache was not rejected")


def test_functionality_uses_umap_from_the_same_cell_cache(tmp_path):
    from alias.evaluation.functionality_benchmark import attach_cached_functionality_umap

    cell_run = tmp_path / "cached_cells"
    umap_path = cell_run / "umap/celltype_label_plots/2026-11-09/df_cells_umap.parquet"
    umap_path.parent.mkdir(parents=True)
    umap_path.write_bytes(b"saved coordinates")
    embeddings = {"MB": {"scrna": {"df_cells": {"run_dir": str(cell_run), "n_samples": 2}}}}
    assert attach_cached_functionality_umap(embeddings, "MB")
    assert embeddings["MB"]["scrna"]["df_cells"]["umap"]["path"] == str(umap_path)
