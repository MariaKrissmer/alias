import importlib.util
import json
from pathlib import Path
import sys
import types

import pandas as pd

from alias.util.artifacts import create_run_directory, save_embedding_frame


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "alias" / "evaluation" / "functionality_cell_similarity.py"
)


def load_functionality_module(monkeypatch):
    fake_similarity_module = types.ModuleType("alias.util.similarity")
    fake_similarity_module.evaluate_similarity = lambda **kwargs: (pd.DataFrame(), None)
    monkeypatch.setitem(sys.modules, "alias.util.similarity", fake_similarity_module)

    fake_plot_module = types.ModuleType("alias.util.plots.umap_plots")

    class DummyPlotter:
        def __init__(self, *args, **kwargs):
            pass

        def plot_similarity_heatmap(self, sim_df, output_path, **kwargs):
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("heatmap", encoding="utf-8")

    fake_plot_module.UMAPCellPlotter = DummyPlotter
    monkeypatch.setitem(sys.modules, "alias.util.plots.umap_plots", fake_plot_module)

    spec = importlib.util.spec_from_file_location(
        "alias.evaluation.functionality_cell_similarity",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_embeddings_dict(tmp_path: Path) -> dict:
    run_dir = create_run_directory(
        root_dir=tmp_path,
        category="embeddings",
        dataset_name="pbmc_3k",
        model_name="demo_model",
        timestamp="2026-03-23T12-34-56",
    )

    cell_df = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], index=pd.Index(["0", "1"], name="cell_id"))
    additional_df = pd.DataFrame([[0.5, 0.6], [0.7, 0.8]], index=pd.Index(["0", "1"], name="cell_id"))

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
        annotation_map={"0": "first functionality", "1": "second functionality"},
        annotation_file_name="df_additional_input_mapping.json",
    )
    additional_meta["run_dir"] = str(run_dir)

    umap_dir = run_dir / "umap" / "celltype_label_plots" / "2026-03-23T12-34-56"
    umap_dir.mkdir(parents=True, exist_ok=True)
    cell_umap = pd.DataFrame(
        {"UMAP1": [0.0, 1.0], "UMAP2": [0.0, 1.0]},
        index=pd.Index(["0", "1"], name="cell_id"),
    )
    cell_umap_path = umap_dir / "df_cells_umap.parquet"
    cell_umap.to_parquet(cell_umap_path)
    cell_meta["umap"] = {"path": str(cell_umap_path), "n_points": 2}

    return {
        "demo_model": {
            "pbmc_3k": {
                "df_cells": cell_meta,
                "df_additional": additional_meta,
            }
        }
    }


def test_functionality_similarity_writes_timestamped_results_run(tmp_path: Path, monkeypatch):
    module = load_functionality_module(monkeypatch)

    def fake_evaluate_similarity(**kwargs):
        return (
            pd.DataFrame(
                {
                    "other_embedding": [
                        "first functionality",
                        "first functionality",
                        "second functionality",
                        "second functionality",
                    ],
                    "ground_truth_column": ["B_cell", "T_cell", "B_cell", "T_cell"],
                    "roc_auc": [0.3, 0.9, 0.4, 0.8],
                }
            ),
            None,
        )

    monkeypatch.setattr(module, "evaluate_similarity", fake_evaluate_similarity)

    results = module.functionality_similarity(
        embeddings_dict=build_embeddings_dict(tmp_path),
        annotation_column="celltype",
        config=module.FunctionalitySimilarityConfig(output_dir=tmp_path / "evaluation_plots"),
    )

    run_dir = (
        tmp_path
        / "evaluation_plots"
        / "demo_model"
        / "pbmc_3k"
        / "functionality_similarity"
        / "2026-03-23T12-34-56"
    )
    assert run_dir.exists()
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "results_df.csv").exists()
    assert (run_dir / "functionality_heatmap.pdf").exists()
    assert "evaluation_plots/evaluation_plots" not in str(run_dir)
    assert list(results["model"]) == ["demo_model"] * 4

    with (run_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["run_timestamp"] == "2026-03-23T12-34-56"
    assert metadata["evaluation_name"] == "functionality_similarity"


def test_functionality_similarity_uses_new_run_directory_on_rerun(tmp_path: Path, monkeypatch):
    module = load_functionality_module(monkeypatch)

    def fake_evaluate_similarity(**kwargs):
        return (
            pd.DataFrame(
                {
                    "other_embedding": ["first functionality", "second functionality"],
                    "ground_truth_column": ["B_cell", "T_cell"],
                    "roc_auc": [0.3, 0.8],
                }
            ),
            None,
        )

    monkeypatch.setattr(module, "evaluate_similarity", fake_evaluate_similarity)

    embeddings_dict = build_embeddings_dict(tmp_path)
    config = module.FunctionalitySimilarityConfig(output_dir=tmp_path / "evaluation_plots")

    module.functionality_similarity(
        embeddings_dict=embeddings_dict,
        annotation_column="celltype",
        config=config,
    )
    module.functionality_similarity(
        embeddings_dict=embeddings_dict,
        annotation_column="celltype",
        config=config,
    )

    function_root = (
        tmp_path
        / "evaluation_plots"
        / "demo_model"
        / "pbmc_3k"
        / "functionality_similarity"
    )
    run_names = sorted(path.name for path in function_root.iterdir() if path.is_dir())
    assert run_names == ["2026-03-23T12-34-56", "2026-03-23T12-34-56_01"]
