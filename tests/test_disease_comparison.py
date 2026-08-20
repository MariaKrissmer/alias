import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd

from alias.util.artifacts import create_run_directory, save_embedding_frame


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "alias" / "evaluation" / "disease_comparison.py"
)


def load_disease_module(monkeypatch):
    fake_scanpy = types.ModuleType("scanpy")
    fake_scanpy.read_h5ad = lambda path: None
    fake_scanpy.settings = types.SimpleNamespace(figdir=".")
    fake_scanpy.tl = types.SimpleNamespace(rank_genes_groups=lambda *args, **kwargs: None)
    fake_scanpy.get = types.SimpleNamespace(rank_genes_groups_df=lambda *args, **kwargs: pd.DataFrame())
    fake_scanpy.pl = types.SimpleNamespace(rank_genes_groups_heatmap=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "scanpy", fake_scanpy)

    spec = importlib.util.spec_from_file_location(
        "alias.evaluation.disease_comparison",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DummyPlotter:
    def __init__(self, *args, **kwargs):
        self.annotate_centroids = False

    def _touch(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("plot", encoding="utf-8")

    def plot_cells(self, df, output_path, title, annotation_column=None, annotate_centroids_df=None, continuous_color_column=None):
        self._touch(output_path)

    def plot_distribution_difference(
        self,
        df,
        x_column,
        y_column,
        label_column,
        pval_column,
        count_column,
        size_scale,
        output_path,
        **kwargs,
    ):
        self._touch(output_path)


class FakeAdataSlice:
    def __init__(self, obs: pd.DataFrame, obsm: dict[str, np.ndarray]):
        self.obs = obs
        self.obsm = obsm
        self.uns = {}
        self.obs_names = obs.index

    def copy(self):
        return FakeAdataSlice(self.obs.copy(), {key: value.copy() for key, value in self.obsm.items()})


class FakeAdata(FakeAdataSlice):
    def copy(self):
        return FakeAdata(self.obs.copy(), {key: value.copy() for key, value in self.obsm.items()})

    def __getitem__(self, mask):
        if isinstance(mask, tuple):
            mask = mask[0]
        if hasattr(mask, "values"):
            mask = mask.values
        return FakeAdataSlice(
            self.obs.loc[self.obs.index[mask]].copy(),
            {key: value[mask].copy() for key, value in self.obsm.items()},
        )


def build_embeddings_dict(tmp_path: Path) -> dict:
    run_dir = create_run_directory(
        root_dir=tmp_path,
        category="embeddings",
        dataset_name="scrna",
        model_name="demo_model",
        timestamp="2026-04-01T10-00-00",
    )

    cell_df = pd.DataFrame(
        [[1.0, 0.0], [0.9, 0.1], [0.1, 0.9], [0.0, 1.0]],
        index=pd.Index(["c1", "c2", "c3", "c4"], name="cell_id"),
    )
    centroid_df = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=pd.Index(["T_cell", "B_cell"], name="cell_id"),
    )

    cell_meta = save_embedding_frame(
        run_dir,
        "df_cells",
        cell_df,
        annotation_map={"celltype": {"c1": "T_cell", "c2": "T_cell", "c3": "B_cell", "c4": "B_cell"}},
    )
    cell_meta["run_dir"] = str(run_dir)

    centroid_meta = save_embedding_frame(
        run_dir,
        "df_celltypes",
        centroid_df,
        annotation_map={"T_cell": "T_cell", "B_cell": "B_cell"},
    )
    centroid_meta["run_dir"] = str(run_dir)

    cell_umap = pd.DataFrame({"UMAP1": [0.0, 1.0, 2.0, 3.0], "UMAP2": [0.0, 1.0, 2.0, 3.0]}, index=cell_df.index)
    centroid_umap = pd.DataFrame({"cell_type": ["T_cell", "B_cell"], "UMAP1": [0.5, 2.5], "UMAP2": [0.5, 2.5]})
    umap_dir = run_dir / "umap"
    umap_dir.mkdir()
    cell_umap_path = umap_dir / "df_cells_umap.parquet"
    centroid_umap_path = umap_dir / "df_celltypes_umap.parquet"
    cell_umap.to_parquet(cell_umap_path)
    centroid_umap.to_parquet(centroid_umap_path)
    cell_meta["umap"] = {"path": str(cell_umap_path)}
    centroid_meta["umap"] = {"path": str(centroid_umap_path)}

    return {"demo_model": {"scrna": {"df_cells": cell_meta, "df_celltypes": centroid_meta}}}


def build_adata() -> FakeAdata:
    obs = pd.DataFrame(
        {
            "celltype": ["T_cell", "T_cell", "B_cell", "B_cell"],
            "subject.cmv": ["Positive", "Negative", "Positive", "Negative"],
        },
        index=["c1", "c2", "c3", "c4"],
    )
    obsm = {"X_umap": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=float)}
    return FakeAdata(obs=obs, obsm=obsm)


def test_disease_comparison_writes_results_run(tmp_path: Path, monkeypatch):
    module = load_disease_module(monkeypatch)

    monkeypatch.setattr(module, "UMAPCellPlotter", DummyPlotter)
    monkeypatch.setattr(module, "_run_deg_analysis", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "load_embedding_model", lambda _: types.SimpleNamespace(encode=lambda texts: np.array([[1.0, 0.0]])))

    def fake_evaluate_similarity_meta(df_cells, df_centroids, out_dir, *, disease_emb, label_key, bins, similarity_metric, **kwargs):
        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "cell_type": kwargs["annotation_column_value"],
            "label": label_key,
            "mw_stat": 1.0,
            "mw_p": 0.01,
            "mean_sim_disease": 0.8,
            "mean_sim_other": 0.2,
            "best_thresh": 0.5,
            "df_sim": df_cells.assign(associated=[True, False]),
        }

    monkeypatch.setattr(module, "evaluate_similarity_meta", fake_evaluate_similarity_meta)

    results = module.disease_comparison(
        embeddings_dict=build_embeddings_dict(tmp_path),
        eval_data={"adata": build_adata()},
        subfolder_fig_dir=tmp_path / "evaluation_plots",
        annotation_column="celltype",
        evaluation_config=module.EvaluationConfig(n_neighbors=5),
        config=module.DiseaseComparisonConfig(),
    )

    run_dir = (
        tmp_path
        / "evaluation_plots"
        / "demo_model"
        / "scrna"
        / "disease_comparison"
        / "2026-04-01T10-00-00"
    )
    assert run_dir.exists()
    assert (run_dir / "metadata.json").exists()
    assert (run_dir / "cells_colored_by_celltype_embeddings.pdf").exists()
    assert (run_dir / "cells_colored_by_disease_positive_adata.pdf").exists()
    assert (run_dir / "disease_only_01" / "results_df.csv").exists()
    assert (run_dir / "disease_only_01" / "distribution_difference_summary.pdf").exists()
    assert set(results["cell_type"]) == {"T_cell", "B_cell"}

    with (run_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["evaluation_name"] == "disease_comparison"
    assert metadata["run_timestamp"] == "2026-04-01T10-00-00"
