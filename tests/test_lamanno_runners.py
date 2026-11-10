from __future__ import annotations

from argparse import Namespace
import importlib.util
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_DIR = ROOT / "scripts/revision1_v1/LaManno/evaluation"


def _load_runner(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _small_adata() -> ad.AnnData:
    return ad.AnnData(
        X=np.zeros((2, 1)),
        obs=pd.DataFrame(
            {"celltype": ["Forebrain", "Forebrain"], "time": [7.0, 8.0], "sample": ["a", "b"]},
            index=["cell_1", "cell_2"],
        ),
    )


def test_s9_runner_generates_released_model_embeddings_inside_current_run(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    run_root = tmp_path / "analysis_runs/current"
    monkeypatch.setenv("ALIAS_REVISION_DATA_ROOT", str(data_root))
    monkeypatch.setenv("LAMANNO_ANALYSIS_RUN_DIR", str(run_root))
    module = _load_runner(
        EVALUATION_DIR / "run_lamanno_embedding_evaluation.py", "test_lamanno_s9_runner"
    )
    s9_dir = data_root / "LaManno/S9_heldout_sample_semantic"
    (s9_dir / "datasets/scrna_test").mkdir(parents=True)
    (s9_dir / "adata_test.h5ad").touch()
    monkeypatch.setattr(module, "load_from_disk", lambda _: object())
    monkeypatch.setattr(module.sc, "read_h5ad", lambda _: _small_adata())
    configs = []

    def generate_embeddings(*, evaluation_dict, embedding_config):
        configs.append(embedding_config)
        return {"ALIASMF": {"scrna": {}}}

    monkeypatch.setattr(module.ev, "generate_embeddings", generate_embeddings)
    monkeypatch.setattr(module.ev, "umap_plots", lambda **kwargs: {})
    options = Namespace(
        batch_size=256, max_cells=50000, force_regenerate=False,
        n_neighbors=50, min_dist=0.5, n_components=50, random_state=73,
        only_timepoint_similarity=False, only_umap=True,
        lineage="lineage_4", cell_origin="Forebrain",
    )
    outputs = module._run_one_spec(module.EVALUATION_SPECS["MO"], options)
    assert configs[0].embedding_models == ["mariakrissmer/ALIAS-MF"]
    assert Path(configs[0].output_dir) == run_root / "embeddings/MO"
    assert Path(outputs["embedding_subset_umaps_dir"]).is_relative_to(run_root)


def test_s10_runner_uses_released_model_and_current_run(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    run_root = tmp_path / "analysis_runs/current"
    monkeypatch.setenv("ALIAS_REVISION_DATA_ROOT", str(data_root))
    monkeypatch.setenv("LAMANNO_ANALYSIS_RUN_DIR", str(run_root))
    module = _load_runner(
        EVALUATION_DIR / "heldout_timepoint_eval.py", "test_lamanno_s10_runner"
    )
    monkeypatch.setattr(module, "load_from_disk", lambda _: object())
    monkeypatch.setattr(module.sc, "read_h5ad", lambda _: _small_adata())
    configs = []

    def generate_embeddings(*, evaluation_dict, embedding_config):
        configs.append(embedding_config)
        return {"ALIASMG": {"scrna": {}}}

    monkeypatch.setattr(module.ev, "generate_embeddings", generate_embeddings)
    monkeypatch.setattr(
        module.ev, "heldout_timepoint_neighbor_enrichment",
        lambda **kwargs: pd.DataFrame({"representation": ["alias"]}),
    )
    options = Namespace(
        batch_size=256, max_cells=50000, force_regenerate=False,
        n_neighbors=50, min_dist=0.5, n_components=50, random_state=73,
        background_alpha=0.12, highlighted_alpha=0.9,
        quantification_k_neighbors=15, only_neighborhood_assessment=True,
    )
    outputs = module._run_one_spec(
        module.EVALUATION_SPECS["MP"], options, heldout_timepoints=["9.0"]
    )
    assert configs[0].embedding_models == ["mariakrissmer/ALIAS-MG"]
    assert Path(configs[0].output_dir) == run_root / "embeddings/MP"
    assert Path(outputs["neighbor_metrics"]).is_relative_to(run_root)
    metadata = json.loads(Path(outputs["metadata"]).read_text(encoding="utf-8"))
    assert metadata["resolved_model_source"] == "mariakrissmer/ALIAS-MG"
