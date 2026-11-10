from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts" / "revision1_v1" / "pbmc3k"


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_pbmc_training_keeps_weights_local(monkeypatch, tmp_path):
    monkeypatch.setenv("PBMC3K_MODEL_ROOT", str(tmp_path))
    module = load_script("pbmc3k_train_local", "run_cell_sentence_length_stability.py")
    run = module.StudyRun(cs_length=50, training_seed=101)

    config = module.make_training_config(run)
    assert config.save_to_local is True
    assert config.save_to_hf is False
    assert Path(config.output_path) == tmp_path / run.model_id

    calls = []
    monkeypatch.setattr(module, "load_triplet_dataset", lambda _: {"scrna": {}})
    monkeypatch.setattr(module, "train_model", lambda **kwargs: calls.append(kwargs))
    module.train_run(run)
    assert calls[0]["hf_upload"] is False


def test_pbmc_evaluation_uses_only_selected_local_weights(monkeypatch, tmp_path):
    monkeypatch.setenv("PBMC3K_MODEL_ROOT", str(tmp_path))
    module = load_script("pbmc3k_eval_local", "evaluate_cell_sentence_length_stability.py")
    specs = module.select_model_specs(module.build_model_specs(), cs_length=50, training_seed=101)

    assert len(specs) == 1
    assert specs[0].model_reference == str(tmp_path / "pbmc3k_cs50_seed101")
    with pytest.raises(FileNotFoundError, match="pbmc3k_cs50_seed101"):
        module.require_local_models(specs)
